import gc
import math
import os
import shutil
import sys
import time
import warnings
from functools import partial

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import dist
from utils import arg_util, misc
from utils.ckpt_compat import pick_dict, strip_prefix
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.lr_control import lr_wd_annealing
from utils.misc import auto_resume


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _unwrap_model(model):
    while hasattr(model, 'module') or hasattr(model, '_orig_mod'):
        model = model.module if hasattr(model, 'module') else model._orig_mod
    return model


def save_var_snapshot(trainer, path, args, step, epoch):
    """Save model weights without optimizer/VAE state for validation and ablation."""
    torch.save({
        'epoch': epoch, 'iter': step,
        'trainer': {'var_wo_ddp': _unwrap_model(trainer.var_wo_ddp).state_dict()},
        'args': args.state_dict(),
    }, path)
    print(f'[saving model snapshot] path={path}', flush=True)


def save_full_checkpoint(trainer, path, args, epoch, last_step):
    """Atomically replace the latest resumable model+optimizer checkpoint."""
    temporary_path = path + '.tmp'
    torch.save({
        'epoch': epoch, 'iter': last_step - 1, 'next_step': last_step,
        'trainer': trainer.state_dict(), 'args': args.state_dict(),
    }, temporary_path)
    os.replace(temporary_path, path)
    print(f'[saving full checkpoint] next_step={last_step} path={path}', flush=True)


def load_pretrained_var(var, state_dict, new_prefixes=()):
    """Load VAR weights; only parameters of newly added modules may be missing."""
    model_sd = var.state_dict()
    for key in [k for k, v in model_sd.items() if k in state_dict and state_dict[k].shape != v.shape]:
        print(f'[load VAR] drop shape-mismatched {key}')
        state_dict.pop(key)
    missing, unexpected = var.load_state_dict(state_dict, strict=False)
    print(f'[load VAR] missing:  {missing}')
    print(f'[load VAR] unexpected:  {unexpected}')
    allowed = {k for k in model_sd if k.startswith(tuple(new_prefixes))} if new_prefixes else set()
    if set(missing) - allowed or unexpected:
        raise RuntimeError(f'Pretrained VAR load mismatch outside {new_prefixes}: '
                           f'missing={sorted(set(missing) - allowed)[:8]}, unexpected={unexpected[:8]}')
    return var


def load_vae_from_ckpt(vae, ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu')
    vae_sd = pick_dict(ck.get('trainer', {}), 'vae_local', 'vae') or pick_dict(ck, 'vae_local', 'vae', 'state_dict', 'model')
    assert vae_sd is not None, f'VAE weights not found in {ckpt_path}'
    vae_sd, cur = strip_prefix(vae_sd), vae.state_dict()
    filtered = {k: w for k, w in vae_sd.items() if k in cur and cur[k].shape == w.shape}
    ret = vae.load_state_dict(filtered, strict=False)
    if ret.missing_keys or ret.unexpected_keys or len(filtered) != len(cur):
        raise RuntimeError(f'Incomplete VAE load from {ckpt_path}: loaded={len(filtered)}/{len(cur)}, '
                           f'missing={ret.missing_keys[:8]}, unexpected={ret.unexpected_keys[:8]}')
    print(f'[load VAE] src={ckpt_path}  loaded={len(filtered)}/{len(cur)}')
    return vae


RESUME_CRITICAL_KEYS = (
    'pn', 'patch_size', 'depth', 'saln', 'anorm', 'fuse',
    'ini', 'hd', 'aln', 'alng',
    'batch_size', 'glb_batch_size', 'ac', 'ep',
    'train_data_path', 'valid_data_path', 'vae_model_path', 'var_pretrain_path',
    'seed', 'same_seed_for_all_ranks', 'head_lr_scale', 'condition_lr_scale',
    'stage_loss_power', 'diff_loss_w', 'img_loss_w', 'l1_w', 'ssim_w', 'ste_tau', 'ste_topk',
    'tblr', 'twd', 'twde', 'tclip', 'wp', 'wp0', 'wpe', 'sche', 'ls',
    'opt', 'afuse', 'fp16', 'tf32',
)


def build_everything(args: arg_util.Args):
    auto_resume_info, _, next_step, trainer_state, args_state = auto_resume(args, 'ar-ckpt-last.pth')
    for key in RESUME_CRITICAL_KEYS if args_state else ():
        saved = args_state.get(key, getattr(arg_util.Args, key))   # options added later count as their default
        if saved != getattr(args, key):
            raise RuntimeError(f'Resume configuration mismatch for {key}: checkpoint={saved!r}, current={getattr(args, key)!r}')
    dist.barrier()
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')

    # data
    print('[build PT data] ...\n')
    dataset_train, dataset_val = build_dataset(args.train_data_path, args.valid_data_path, final_reso=args.data_load_reso)
    types = str((type(dataset_train).__name__, type(dataset_val).__name__))
    expected_iters_train = math.ceil(len(dataset_train) / args.glb_batch_size)
    start_ep, start_it = divmod(next_step, expected_iters_train)
    if start_ep >= args.ep:
        raise RuntimeError(f'Run already completed {next_step} steps for ep={args.ep}')
    ld_val = DataLoader(
        dataset_val, num_workers=min(args.workers, 2), pin_memory=False,
        batch_size=min(args.batch_size, 2), sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
        shuffle=False, drop_last=False,
    )
    ld_train = DataLoader(
        dataset=dataset_train, num_workers=args.workers, pin_memory=False,
        generator=args.get_different_generator_for_each_rank(),
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size, same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(), start_ep=start_ep, start_it=start_it,
        ),
    )
    del dataset_train, dataset_val
    [print(line) for line in auto_resume_info]
    iters_train = len(ld_train)
    assert iters_train == expected_iters_train
    ld_train = iter(ld_train)
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}')

    # models
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import build_var
    from trainer import VARTrainer
    from utils.amp_sc import AmpOptimizer
    from utils.lr_control import filter_params

    vae_local, var_wo_ddp = build_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,       # hard-coded VQVAE hyperparameters
        device=dist.get_device(), patch_nums=args.patch_nums,
        depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
    )
    dist.barrier()
    # The VAE stays frozen: encoder and codebook give the targets, the decoder serves the pixel losses.
    vae_local = load_vae_from_ckpt(vae_local, args.vae_model_path)
    for p in vae_local.parameters():
        p.requires_grad_(False)
    vae_local.quantize.eval()
    vae_local.encoder.eval()
    vae_local.decoder.to(dtype=torch.float32)
    vae_local.post_quant_conv.to(dtype=torch.float32)

    var_state = torch.load(args.var_pretrain_path, map_location='cpu')['trainer']['var_wo_ddp']
    var_wo_ddp = load_pretrained_var(var_wo_ddp, var_state)
    del var_state
    var_wo_ddp.ste_tau = float(args.ste_tau)
    var_wo_ddp.ste_topk = int(args.ste_topk)

    # A frozen VAE is not wrapped by DDP (no trainable parameters); each rank keeps a local copy.
    vae = NullDDP(vae_local)
    var = (DDP if dist.initialized() else NullDDP)(
        var_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False,
        gradient_as_bucket_view=True, broadcast_buffers=False, static_graph=True,
    )
    print(f'[INIT] VAR model = {var_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}'
    print('[INIT][#para] ' + ', '.join(f'{k}={count_p(m)}' for k, m in (
        ('VAE', vae_local), ('VAE.enc', vae_local.encoder), ('VAE.dec', vae_local.decoder), ('VAE.quant', vae_local.quantize), ('VAR', var_wo_ddp))) + '\n\n')

    # optimizer
    lr_scales = {
        'head.': args.head_lr_scale, 'head_nm.': args.head_lr_scale, 'decoder_norm.': args.head_lr_scale,
        'con_embedding.': args.condition_lr_scale,
    }
    names, paras, para_groups = filter_params(var_wo_ddp, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta', 'ada_gss', 'moe_bias', 'scale_mul', 'diffloss',
    }, lr_scales=lr_scales)
    opt_clz = {
        'adam': partial(torch.optim.Adam, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}, #groups={len(para_groups)}')
    var_optim = AmpOptimizer(
        mixed_precision=args.fp16, optimizer=opt_clz(params=para_groups, **opt_kw), names=names, paras=paras,
        grad_clip=args.tclip, n_gradient_accumulation=args.ac,
    )
    del names, paras, para_groups

    trainer = VARTrainer(
        device=args.device, patch_nums=args.patch_nums, resos=args.resos,
        vae_local=vae, var_wo_ddp=var_wo_ddp, var=var, var_opt=var_optim, label_smooth=args.ls,
        stage_loss_power=args.stage_loss_power, diff_loss_w=args.diff_loss_w,
        img_loss_w=args.img_loss_w, l1_w=args.l1_w, ssim_w=args.ssim_w,
    )
    if trainer_state:
        trainer.load_state_dict(trainer_state, strict=False)
    del vae_local, var_wo_ddp, var, var_optim
    dist.barrier()
    return trainer, start_ep, start_it, iters_train, ld_train, ld_val


def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    trainer, start_ep, start_it, iters_train, ld_train, ld_val = build_everything(args)

    start_time = time.time()
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999., 999., -1., -1.
    best_val_loss_tail = 999
    L_mean, L_tail = -1, -1
    stop_training = False
    for ep in range(start_ep, args.ep):
        me = misc.MetricLogger(delimiter='  ')
        me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
        me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
        [me.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
        for name in ('Lce', 'Ldiff', 'Limg', 'Ll1', 'Lssim', 'Ltot'):
            me.add_meter(name, misc.SmoothedValue(fmt='{median:.4f} ({global_avg:.4f})'))
        header = f'[Ep]: [{ep + 1:4d}/{args.ep}]'
        if ep == start_ep:
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
        g_it, max_it = ep * iters_train, args.ep * iters_train
        if args.max_train_steps > 0:
            max_it = min(max_it, args.max_train_steps)  # a truncated run anneals by its own end
        ep_start_it = start_it if ep == start_ep else 0

        for it, batch in me.log_every(ep_start_it, iters_train, ld_train, 300 if iters_train > 8000 else 50, header):
            g_it = ep * iters_train + it
            if it < ep_start_it:
                continue
            if ep == start_ep and it == ep_start_it:
                warnings.resetwarnings()
            inp = batch['pixel_values'].to(args.device, non_blocking=True)
            lr_inp = batch['conditioning_pixel_values'].to(args.device, non_blocking=True)
            args.cur_it = f'{it + 1}/{iters_train}'
            wp_it = args.wp * iters_train
            min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.var_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
            args.cur_lr, args.cur_wd = max_tlr, max_twd
            stepping = (g_it + 1) % args.ac == 0
            trainer.train_step(it=it, stepping=stepping, metric_lg=me, inp_B3HW=inp, lr_inp=lr_inp)
            me.update(tlr=max_tlr)

            if (args.step_ckpt_interval > 0 and stepping and (g_it + 1) % args.step_ckpt_interval == 0 and (it + 1) < iters_train):
                if dist.is_local_master():
                    save_full_checkpoint(trainer, args.last_ckpt_path, args, ep + 1, g_it + 1)
                dist.barrier()

            if args.max_train_steps > 0 and (g_it + 1) >= args.max_train_steps:
                if dist.is_local_master():
                    save_var_snapshot(trainer, os.path.join(args.local_out_dir_path, f'ar-model-step{g_it + 1}.pth'), args, g_it + 1, ep + 1)
                print(f'[training stop] reached max_train_steps={args.max_train_steps}', flush=True)
                stop_training = True
                break

            if (it + 1) % iters_train == 0:
                # release the last step's references before validation
                del batch, inp, lr_inp
                trainer.var_opt.optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = trainer.eval_ep(ld_val)
                best_updated = best_val_loss_tail > val_loss_tail
                best_val_loss_tail = min(best_val_loss_tail, val_loss_tail)
                args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail
                print(f' [*] [ep{ep}]  (val {tot})  Lm: {val_loss_mean:.4f}, Lt: {val_loss_tail:.4f}, Acc m&t: {val_acc_mean:.2f} {val_acc_tail:.2f},  Val cost: {cost:.2f}s')
                if dist.is_local_master():
                    local_out_ckpt = os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth')
                    local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'ar-ckpt-best.pth')
                    print('[saving ckpt] ...', end='', flush=True)
                    save_full_checkpoint(trainer, local_out_ckpt, args, ep + 1, g_it + 1)
                    if best_updated and args.save_ce_best:
                        shutil.copy(local_out_ckpt, local_out_ckpt_best)
                    if args.epoch_snapshot_interval > 0 and (ep + 1) % args.epoch_snapshot_interval == 0:
                        save_var_snapshot(trainer, os.path.join(args.local_out_dir_path, f'ar-model-ep{ep + 1:03d}.pth'), args, g_it + 1, ep + 1)
                    saved_path = local_out_ckpt_best if best_updated and args.save_ce_best else local_out_ckpt
                    print(f'     [saving ckpt](*) finished!  @ {saved_path}', flush=True, clean=True)
                dist.barrier()

        if stop_training:
            dist.barrier()
            break

        me.synchronize_between_processes()
        stats = {k: meter.global_avg for k, meter in me.meters.items()}
        sec, remain_time, finish_time = me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(best_acc_mean, acc_mean)
        if L_tail != -1:
            best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(best_acc_tail, acc_tail)
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = L_mean, L_tail, acc_mean, acc_tail, grad_norm
        args.cur_ep = f'{ep + 1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        print(f'     [ep{ep}]  (training )  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        args.dump_log()

    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print(f'\n\n  [*] [PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})\n\n')
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    args.remain_time, args.finish_time = '-', time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log()
    dist.barrier()


if __name__ == '__main__':
    try:
        main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
