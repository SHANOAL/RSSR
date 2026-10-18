import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn_RoPE, precompute_freqs_cis
from models.diffusion.diffloss import DiffLoss
from models.helpers import sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class ControlNetConditioningEmbedding(nn.Module):
    """LR encoder: stride-2 conv stages map the upsampled LR image to the prefix-token grid."""

    def __init__(self, conditioning_embedding_channels: int, conditioning_channels: int = 3,
                 block_out_channels: Tuple[int, ...] = (128, 256, 512, 1024)):
        super().__init__()
        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            channel_in, channel_out = block_out_channels[i], block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
        self.conv_out = nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)

    def forward(self, conditioning):
        embedding = F.silu(self.conv_in(conditioning), inplace=True)
        for block in self.blocks:
            embedding = F.silu(block(embedding), inplace=True)
        return self.conv_out(embedding)


class VAR_RoPE(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads

        # Sequence layout: 625 LR prefix tokens and the condition token, then the
        # token maps of scales 2..S (the 1x1 scale is predicted from the condition token).
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[-1] ** 2 + 1
        self.context_token = self.first_l
        self.last_level_pns = self.patch_nums[-1] ** 2
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())

        # 1. LR condition and token embeddings
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.con_embedding = ControlNetConditioningEmbedding(self.C, 3, (32, 128, 256, 512, 1536))
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 2. condition token and start positions
        init_std = math.sqrt(1 / self.C / 3)
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.uncond_token = nn.Parameter(torch.empty(1, self.C))
        nn.init.trunc_normal_(self.uncond_token.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        rope_patch_nums = (self.patch_nums[-1], self.patch_nums[0], self.patch_nums[1]) + self.patch_nums[2:]
        self.freqs_cis = precompute_freqs_cis(self.C // num_heads, rope_patch_nums)

        # level embedding, distinguishing the levels of the token pyramid
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # 3. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6 * self.C)) if shared_aln else nn.Identity()
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn_RoPE(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        print(
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 4. block-causal attention mask for teacher-forced training (inference uses the kv cache)
        d: torch.Tensor = torch.cat(
            [torch.full((self.context_token,), 0)] + [torch.full((pn * pn,), i + 1) for i, pn in enumerate(self.patch_nums[1:])]
        ).view(1, self.L + self.context_token - 1, 1)
        dT = d.transpose(1, 2)
        self.register_buffer('lvl_1L', dT[:, 0].contiguous())
        attn_bias_for_masking = torch.where(d >= dT, 0.0, -torch.inf).reshape(1, 1, self.L + self.context_token - 1, self.L + self.context_token - 1)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # 5. token head and the residual diffusion refiner of the last scale
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
        self.decoder_norm = norm_layer(self.C)
        self.diffloss = DiffLoss(target_channels=self.Cvae, z_channels=self.C, width=1024, depth=6, num_sampling_steps='10', sampler='iddpm')
        self.diffusion_batch_mul = 4
        self.ste_tau = 0.5
        self.ste_topk = 16

    def get_logits(self, h: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    def _encode_condition(self, lr_inp: torch.Tensor):
        return self.con_embedding(lr_inp)

    def forward_diff_loss(self, z, target, mask=None):
        bs, seq_len, _ = target.shape
        target = target.reshape(bs * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bs * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        return self.diffloss(z=z, target=target, mask=mask)

    @torch.no_grad()
    def autoregressive_infer(self, lr_inp: torch.Tensor, top_k=0, top_p=0.0, g_seed: Optional[int] = None, return_stages: bool = False,
                             refine_temperature: float = 0.0):
        """Coarse-to-fine code prediction followed by the residual refiner.

        refine_temperature scales the noise injected at every reverse step of the refiner; 0 gives deterministic
        residual refinement. Returns the final latent (B, Cvae, h, w); with return_stages also the cumulative latent
        after every scale.
        """
        B = lr_inp.shape[0]
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed)
            rng = self.rng

        lr_feat = self._encode_condition(lr_inp)
        lr_tokens = lr_feat.view(B, self.C, -1).permute(0, 2, 1)
        assert lr_tokens.shape[1] == self.context_token - 1
        cond_BD = self.uncond_token.expand(B, -1)
        sos = torch.cat((lr_tokens, cond_BD.unsqueeze(1)), dim=1)

        lvl_pos = self.lvl_embed(self.lvl_1L)
        next_token_map = sos + self.pos_start.expand(B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        stage_f_hats = [] if return_stages else None
        self.freqs_cis = self.freqs_cis.to(dist.get_device())

        for b in self.blocks:
            b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):
            if si > 0:
                freqs_cis = self.freqs_cis[cur_L:cur_L + pn * pn]
                cur_L += pn * pn
            else:
                freqs_cis = self.freqs_cis[0:self.context_token]
                cur_L += self.context_token
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, freqs_cis=freqs_cis, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            if si == 0:
                logits_BlV = logits_BlV[:, [-1], :]
            tk = 1 if (top_k <= 0 and top_p <= 0.0) else top_k
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=tk, top_p=top_p, num_samples=1)[:, :, 0]
            if si == self.num_stages_minus_1:
                last_layer_cond = x
            h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if return_stages:
                # the cumulative latent lives on the final grid, so every snapshot decodes like the final prediction
                stage_f_hats.append(f_hat.detach().clone())
            if si != self.num_stages_minus_1:
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si + 1] ** 2]
        for b in self.blocks:
            b.attn.kv_caching(False)

        # Residual refiner on the last scale. The duplicated batch with cfg=0 is the sampler path
        # (and random stream) behind every reported result, so it is kept as is.
        last_stage_cond = self.decoder_norm(last_layer_cond + self.word_embed(self.vae_quant_proxy[0].embedding(idx_Bl)))
        bs, cur_seq_len, _ = last_stage_cond.shape
        last_stage_cond = last_stage_cond.reshape(bs * cur_seq_len, -1)
        h_BChw_diff = self.diffloss.sample(z=torch.cat([last_stage_cond, last_stage_cond], dim=0), temperature=float(refine_temperature), cfg=0.0)
        h_BChw_diff = h_BChw_diff[:last_stage_cond.shape[0]].reshape(bs, cur_seq_len, -1)
        last_ps = self.patch_nums[-1]
        f_hat += h_BChw_diff.reshape(B, last_ps * last_ps, -1).transpose(1, 2).reshape(B, self.Cvae, last_ps, last_ps)
        return (f_hat, stage_f_hats) if return_stages else f_hat

    def forward(self, x_BLCv_wo_first_l: torch.Tensor, lr_inp: torch.Tensor,
                last_layer_gt: torch.Tensor, last_layer_gt_discrete: torch.Tensor, return_img: bool = True):
        """Teacher-forced pass.

        :param x_BLCv_wo_first_l: teacher-forcing inputs of scales 2..S
        :param last_layer_gt: continuous last-scale latent before its discrete token (B, HW, Cvae)
        :param last_layer_gt_discrete: ground-truth last-scale tokens (B, HW)
        :return: logits of all token maps (B, L, V), the refiner diffusion loss, and the
                 straight-through decoded image in [0, 1] for the pixel losses (None if not requested)
        """
        B = x_BLCv_wo_first_l.shape[0]
        ed = self.L + self.context_token - 1
        HW = self.last_level_pns
        with torch.cuda.amp.autocast(enabled=False):
            lr_feat = self._encode_condition(lr_inp)
            cond_BD = self.uncond_token.expand(B, -1)
            sos = torch.cat((lr_feat.view(B, self.C, -1).permute(0, 2, 1), cond_BD.unsqueeze(1)), dim=1)
            sos = sos.expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1))

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        # the dtype of the mixed-precision context
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        self.freqs_cis = self.freqs_cis.to(x_BLC.device)
        freqs_cis = self.freqs_cis.repeat(B, 1, 1)

        for b in self.blocks:
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, freqs_cis=freqs_cis, attn_bias=attn_bias)
        last_layer_cond = x_BLC[:, ed - HW:, :]
        x_BLC_logits = self.get_logits(x_BLC.float(), cond_BD)
        logits_BLV = x_BLC_logits[:, self.context_token - 1:, :]

        # refiner target: the continuous remainder after the (greedy) last-scale token
        with torch.no_grad():
            try:
                idx_BL_sampled = sample_with_top_k_top_p_(
                    x_BLC_logits[:, ed - HW:].clone().detach(), rng=self.rng, top_k=1, top_p=0.96, num_samples=1,
                )[:, :, 0]
            except Exception:
                idx_BL_sampled = last_layer_gt_discrete
        last_stage_discrete_embed = self.vae_quant_proxy[0].embedding(idx_BL_sampled)
        last_layer_cond = self.decoder_norm(last_layer_cond + self.word_embed(last_stage_discrete_embed))
        diff_loss = self.forward_diff_loss(z=last_layer_cond, target=last_layer_gt - last_stage_discrete_embed)

        img_hat = None
        if return_img:
            # Straight-through last scale: hard codebook vectors forward, the soft top-k
            # expectation backward. Earlier scales come from the teacher-forced prefix.
            last_ps = self.patch_nums[-1]
            embedding = self.vae_quant_proxy[0].embedding
            Cvae = embedding.weight.shape[1]
            logits_last = x_BLC_logits[:, ed - HW:].float()
            V = logits_last.shape[-1]
            codebook = embedding.weight.detach().to(dtype=logits_last.dtype, device=logits_last.device, non_blocking=True)
            if self.ste_topk and 0 < self.ste_topk < V:
                topv, topi = torch.topk(logits_last, k=self.ste_topk, dim=-1)
                prob_top = torch.softmax(topv / self.ste_tau, dim=-1)
                code_sel = codebook.index_select(0, topi.reshape(-1)).view(*topi.shape, -1)
                e_soft = (prob_top.unsqueeze(-1) * code_sel).sum(dim=-2)
            else:
                e_soft = torch.softmax(logits_last / self.ste_tau, dim=-1) @ codebook
            e_hard = embedding(torch.argmax(logits_last, dim=-1)).detach()
            e_pred = e_soft + (e_hard - e_soft).detach()
            prior_fhat = x_BLCv_wo_first_l[:, -HW:, :].transpose(1, 2).reshape(B, Cvae, last_ps, last_ps).detach()
            final_residual = e_pred.view(B, HW, Cvae).transpose(1, 2).reshape(B, Cvae, last_ps, last_ps)
            img_hat = (self.vae_proxy[0].fhat_to_img(prior_fhat + final_residual) + 1) * 0.5
        return logits_BLV, diff_loss, img_hat

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0:
            init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight:
                    m.weight.data.fill_(1.)
                if with_bias:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias:
                    m.bias.data.zero_()

        if init_head >= 0:
            self.head.weight.data.mul_(init_head)
            self.head.bias.data.zero_()
        self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
        self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)
        for sab in self.blocks:
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if getattr(sab.ffn, 'fcg', None) is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2 * self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2 * self.C].mul_(init_adaln_gamma)
                if sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'
