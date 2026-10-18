import time
from typing import Optional, Tuple

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dist
from models import VQVAE, VectorQuantizer2
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger


def _tofloat(x):
    return float(x.detach().mean().item()) if torch.is_tensor(x) else float(x)


def l1_loss(pred, gt):
    return F.l1_loss(pred, gt, reduction='none').mean()


def ssim_loss(pred, gt):
    return kornia.losses.ssim_loss(pred, gt, window_size=5, reduction='none').mean(dim=(-1, -2, -3)).mean()


class VARTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local, var_wo_ddp, var, var_opt: AmpOptimizer, label_smooth: float,
        stage_loss_power: float = 1.0, diff_loss_w: float = 2.0,
        img_loss_w: float = 0.0, l1_w: float = 1.0, ssim_w: float = 0.0,
    ):
        super(VARTrainer, self).__init__()
        self.var = var
        self.vae_local = vae_local          # may be wrapped by NullDDP
        self.vae_core: VQVAE = vae_local.module if hasattr(vae_local, 'module') else vae_local
        self.quantize_local: VectorQuantizer2 = self.vae_core.quantize
        self.var_wo_ddp = var_wo_ddp
        self.var_opt = var_opt
        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)

        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.start_token = self.L - patch_nums[-1] * patch_nums[-1]
        self.last_grad_norm = 0.0
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for pn in patch_nums:
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn

        self.stage_loss_power = float(stage_loss_power)
        self.diff_loss_w = float(diff_loss_w)
        self.img_loss_w = float(img_loss_w)
        self.l1_w = float(l1_w)
        self.ssim_w = float(ssim_w)
    def _reduce_stage_ce(self, ce_BL: torch.Tensor) -> torch.Tensor:
        """Average CE within each scale, then apply normalized scale weights."""
        spans = self.begin_ends
        terms = [ce_BL[:, bg:ed].mean() for bg, ed in spans]
        weights = ce_BL.new_tensor([float(ed - bg) ** self.stage_loss_power for bg, ed in spans])
        weights = weights / weights.sum().clamp_min(1e-12)
        return (torch.stack(terms) * weights).sum()

    def _tokenize(self, inp_B3HW):
        f = self.vae_core.quant_conv(self.vae_core.encoder(inp_B3HW))
        gt_idx_Bl, idx_N_list = self.quantize_local.f_to_idxBl_or_fhat(f, to_fhat=False)
        gt_BL = torch.cat(gt_idx_Bl[0:len(self.patch_nums)], dim=1)
        x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(idx_N_list)
        return gt_idx_Bl, gt_BL, x_BLCv_wo_first_l

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training_var, training_vae = self.var_wo_ddp.training, self.vae_core.training
        self.var_wo_ddp.eval()
        self.vae_core.eval()
        for batch in ld_val:
            inp_B3HW = batch['pixel_values'].to(dist.get_device(), non_blocking=True)
            lr_inp = batch['conditioning_pixel_values'].to(dist.get_device(), non_blocking=True)
            B, V = inp_B3HW.shape[0], self.vae_core.vocab_size
            gt_idx_Bl, gt_BL, x_BLCv_wo_first_l = self._tokenize(inp_B3HW)
            logits_BLV, _, _ = self.var(x_BLCv_wo_first_l, lr_inp, last_layer_gt=gt_idx_Bl[-1],
                                        last_layer_gt_discrete=gt_idx_Bl[-2], return_img=False)
            L_mean += self.val_loss(logits_BLV.contiguous().data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, self.start_token:].reshape(-1, V), gt_BL[:, self.start_token:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100 / gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, self.start_token:].argmax(dim=-1) == gt_BL[:, self.start_token:]).sum() * (100 / (gt_BL.shape[1] - self.start_token))
            tot += B
        self.var_wo_ddp.train(training_var)
        self.vae_core.train(training_vae)
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time() - stt

    def train_step(self, it: int, stepping: bool, metric_lg: MetricLogger, inp_B3HW, lr_inp) -> Tuple[Optional[float], Optional[float]]:
        B, V = inp_B3HW.shape[0], self.vae_core.vocab_size
        with torch.no_grad():
            gt_idx_Bl, gt_BL, x_BLCv_wo_first_l = self._tokenize(inp_B3HW)

        with self.var_opt.amp_ctx:
            logits_BLV, diff_loss, img_hat = self.var(x_BLCv_wo_first_l, lr_inp, last_layer_gt=gt_idx_Bl[-1],
                                                      last_layer_gt_discrete=gt_idx_Bl[-2], return_img=self.img_loss_w > 0)
            logits_loss = self._reduce_stage_ce(self.train_loss(logits_BLV.contiguous().view(-1, V), gt_BL.view(-1)).view(B, -1))
            loss = logits_loss + diff_loss * self.diff_loss_w
            logs = {'Lce': _tofloat(logits_loss), 'Ldiff': _tofloat(diff_loss)}
            if self.img_loss_w > 0:
                if img_hat is None or not img_hat.requires_grad:
                    raise RuntimeError('image reconstruction loss has no gradient path to VAR')
                img_gt = (inp_B3HW + 1) * 0.5
                img_loss = torch.tensor(0.0, device=inp_B3HW.device, dtype=img_hat.dtype)
                if self.l1_w > 0:
                    l1 = self.l1_w * l1_loss(img_hat, img_gt)
                    img_loss = img_loss + l1
                    logs['Ll1'] = _tofloat(l1)
                if self.ssim_w > 0:
                    ssim = ssim_loss(img_hat, img_gt)
                    img_loss = img_loss + self.ssim_w * ssim
                    logs['Lssim'] = _tofloat(ssim)
                loss = loss + self.img_loss_w * img_loss
                logs['Limg'] = _tofloat(img_loss)
            logs['Ltot'] = _tofloat(loss)

        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        if grad_norm is not None:  # None on the accumulation micro-batches that do not step
            self.last_grad_norm = grad_norm

        if it == 0 or it in metric_lg.log_iters:
            pred_BL = logits_BLV.data.argmax(dim=-1)
            Lmean = self.val_loss(logits_BLV.contiguous().data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            Ltail = self.val_loss(logits_BLV.data[:, self.start_token:].reshape(-1, V), gt_BL[:, self.start_token:].reshape(-1)).item()
            acc_tail = (pred_BL[:, self.start_token:] == gt_BL[:, self.start_token:]).float().mean().item() * 100
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=float(self.last_grad_norm), **logs)
        return grad_norm, scale_log2

    def get_config(self):
        return {'patch_nums': self.patch_nums, 'resos': self.resos, 'label_smooth': self.label_smooth}

    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if hasattr(m, '_orig_mod'):
                m = m._orig_mod
            state[k] = m.state_dict()
        return state

    def load_state_dict(self, state, strict=True):
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if hasattr(m, '_orig_mod'):
                m = m._orig_mod
            ret = m.load_state_dict(state[k], strict=strict)
            if ret is not None:
                missing, unexpected = ret
                print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        config = state.pop('config', None) or {}
        for k, v in self.get_config().items():
            if config.get(k, None) != v:
                err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                if strict:
                    raise AttributeError(err)
                print(err)
