"""Read model settings from a checkpoint and load VAR and VAE weights strictly."""
import math
from typing import Dict, Optional, Sequence, Tuple

import torch

KNOWN_PATCH_NUMS: Tuple[Tuple[int, ...], ...] = ((1, 2, 3, 4, 6, 9, 13, 18, 25),)


def strip_prefix(sd, prefixes=('module.', 'vae_local.', 'vae.', 'model.', 'net.', 'generator.')):
    out = {}
    for k, v in sd.items():
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
        out[k] = v
    return out


def pick_dict(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d and isinstance(d[k], dict):
            return d[k]
    return None


def parse_patch_nums(raw) -> Optional[Tuple[int, ...]]:
    if raw is None:
        return None
    try:
        if isinstance(raw, (tuple, list)):
            parsed = tuple(int(x) for x in raw)
        else:
            parsed = tuple(int(p) for p in str(raw).strip().replace('-', '_').split('_') if p)
    except ValueError:
        return None
    return parsed or None


def _infer_patch_nums_from_shapes(var_sd: Dict[str, torch.Tensor]) -> Optional[Tuple[int, ...]]:
    lvl, pos_start, attn_bias = var_sd.get('lvl_embed.weight'), var_sd.get('pos_start'), var_sd.get('attn_bias_for_masking')
    if lvl is None or pos_start is None or attn_bias is None:
        return None
    last_sq = int(pos_start.shape[1]) - 1
    last_pn = int(round(math.sqrt(max(last_sq, 0))))
    total_L = int(attn_bias.shape[-1]) - last_sq
    for cand in KNOWN_PATCH_NUMS:
        if len(cand) == int(lvl.shape[0]) and cand[-1] == last_pn and sum(p * p for p in cand) == total_L:
            return cand
    return None


def load_var_checkpoint_and_cfg(var_ckpt_path: str, fallback_patch_nums: Optional[Sequence[int]] = None, verbose: bool = True):
    """Return (checkpoint, VAR state dict, model settings) with the modules enabled in the checkpoint."""
    ck = torch.load(var_ckpt_path, map_location='cpu')
    trainer = ck.get('trainer', {}) if isinstance(ck, dict) else {}
    var_sd_raw = pick_dict(trainer, 'var_wo_ddp') or pick_dict(ck, 'var_wo_ddp', 'var', 'state_dict', 'model')
    if var_sd_raw is None:
        raise RuntimeError(f'VAR weights not found in {var_ckpt_path}')
    var_sd = strip_prefix(var_sd_raw)

    args_obj = ck.get('args', None) if isinstance(ck, dict) else None
    args_dict = args_obj if isinstance(args_obj, dict) else (vars(args_obj) if hasattr(args_obj, '__dict__') else {})
    patch_nums = parse_patch_nums(args_dict.get('pn')) or _infer_patch_nums_from_shapes(var_sd)
    if patch_nums is None:
        if fallback_patch_nums is None:
            raise RuntimeError(f'Cannot infer patch nums of {var_ckpt_path}')
        patch_nums = tuple(int(x) for x in fallback_patch_nums)
        print(f'[WARN] Failed to infer patch nums from {var_ckpt_path}; fallback={patch_nums}')
    depth = 1 + max(int(k.split('.')[1]) for k in var_sd if k.startswith('blocks.') and k.split('.')[1].isdigit())

    cfg = {'patch_nums': tuple(int(x) for x in patch_nums), 'depth': depth}
    if verbose:
        print(f"[INFO] model cfg from the checkpoint: patch_nums={cfg['patch_nums']} depth={cfg['depth']}")
    return ck, var_sd, cfg


def load_var_and_vae(vae, var, var_ckpt_path, vae_pretrain_path=None, ck=None, var_sd=None, verbose=True):
    """Strictly load VAR weights, and VAE weights from the checkpoint or from vae_pretrain_path."""
    ck = ck if ck is not None else torch.load(var_ckpt_path, map_location='cpu')
    trainer = ck.get('trainer', {}) if isinstance(ck, dict) else {}
    var_sd = var_sd if var_sd is not None else (pick_dict(trainer, 'var_wo_ddp') or pick_dict(ck, 'var_wo_ddp', 'var', 'state_dict', 'model'))
    if var_sd is None:
        raise RuntimeError(f'VAR weights not found in {var_ckpt_path}')
    missing, unexpected = var.load_state_dict(strip_prefix(var_sd), strict=True)
    if verbose:
        print(f'[load VAR] strict=True missing={len(missing)} unexp={len(unexpected)}')

    vae_sd = pick_dict(trainer, 'vae_local', 'vae') or pick_dict(ck, 'vae_local', 'vae')
    vae_src = var_ckpt_path
    if vae_sd is None:
        if vae_pretrain_path is None:
            raise RuntimeError('No VAE weights in var_ckpt and vae_pretrain_path is not provided')
        ck_pre = torch.load(vae_pretrain_path, map_location='cpu')
        tr_pre = ck_pre.get('trainer', {}) if isinstance(ck_pre, dict) else {}
        vae_sd = pick_dict(tr_pre, 'vae_local', 'vae') or pick_dict(ck_pre, 'vae_local', 'vae', 'state_dict', 'model')
        if vae_sd is None:
            raise RuntimeError(f'VAE weights not found in {vae_pretrain_path}')
        vae_src = vae_pretrain_path
    missing, unexpected = vae.load_state_dict(strip_prefix(vae_sd), strict=True)
    if verbose:
        print(f'[load VAE] strict=True missing={len(missing)} unexp={len(unexpected)}')
        print(f'[INFO] VAE source: {vae_src}')

    vae.eval()
    var.eval()
