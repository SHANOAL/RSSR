import json
import os
import random
import subprocess
import sys
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import torch
from tap import Tap

import dist


def _git(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return ''


class Args(Tap):
    train_data_path: str = 'datasets/train'
    valid_data_path: str = 'datasets/val'
    vae_model_path: str = 'checkpoints/vqvae.pth'     # frozen VQ autoencoder
    var_pretrain_path: str = 'checkpoints/init.pth'   # transformer initialization for training
    var_test_path: str = 'checkpoints/crossvar.pth'   # trained CrossVAR weights for inference
    exp_name: str = 'crossvar'
    test_root: str = ''         # inference: folder with LR/ (and HR/)
    refine_temperature: float = 0.0   # inference: noise scale of the residual refiner, 0 = deterministic residual refinement
    fusion: str = 'rgb'         # inference: rgb | latent_window | latent | none
    overlap: int = 5            # inference: window overlap in latent positions, 5 starts every window on the input grid
    export_stages: str = ''     # inference: also save the decoded latent after every scale to this folder
    report_json: str = ''       # inference: write timing and memory statistics to this file

    # model
    depth: int = 24
    saln: bool = False          # shared adaln
    anorm: bool = True          # L2 normalized attention
    fuse: bool = False          # fused ops such as flash attn and fused MLP
    ini: float = -1             # -1: automated parameter initialization
    hd: float = 0.02            # head.w *= hd
    aln: float = 0.5            # multiplier of ada_lin.w's initialization
    alng: float = 1e-5          # multiplier of ada_lin.w[gamma channels]'s initialization

    # learning-rate multipliers of the output head and the condition encoder
    head_lr_scale: float = 1.0
    condition_lr_scale: float = 1.0

    # checkpoints
    epoch_snapshot_interval: int = 0
    step_ckpt_interval: int = 0
    save_ce_best: bool = True
    max_train_steps: int = 0

    # training objectives
    stage_loss_power: float = 1.0
    diff_loss_w: float = 2.0
    img_loss_w: float = 1.0
    l1_w: float = 0.0
    ssim_w: float = 0.0
    ste_tau: float = 1.0
    ste_topk: int = 16
    ls: float = 0.0             # label smoothing

    # optimization
    fp16: int = 2               # 1: fp16, 2: bf16
    tblr: float = 1e-5          # base lr
    tlr: float = None           # [automatically set] lr = ac * tblr
    twd: float = 0.05           # initial wd
    twde: float = 0.05          # final wd
    tclip: float = 2.           # <=0 for no grad clip
    bs: int = 4                 # [automatically set] global batch size
    batch_size: int = 4         # batch size per GPU
    glb_batch_size: int = 0     # [automatically set] batch_size * world_size
    ac: int = 1                 # gradient accumulation
    ep: int = 100
    wp: float = 0.1             # warm-up epochs
    wp0: float = 0.005          # initial lr ratio of the warm-up
    wpe: float = 0.01           # final lr ratio
    sche: str = 'lin0'          # lr schedule
    opt: str = 'adamw'
    afuse: bool = False         # fused adamw

    # data
    pn: str = '1_2_3_4_6_9_13_18_25'
    patch_size: int = 16
    patch_nums: tuple = None    # [automatically set] from pn
    resos: tuple = None         # [automatically set] pn * patch_size
    data_load_reso: int = None  # [automatically set] max(resos)
    workers: int = 24

    # [automatically set] run information and logs
    cmd: str = ' '.join(sys.argv[1:])
    branch: str = _git('git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD') or '[unknown]'
    commit_id: str = _git('git rev-parse HEAD') or '[unknown]'
    commit_msg: str = (_git('git log -1').splitlines() or ['[unknown]'])[-1].strip()
    acc_mean: float = None
    acc_tail: float = None
    L_mean: float = None
    L_tail: float = None
    vacc_mean: float = None
    vacc_tail: float = None
    vL_mean: float = None
    vL_tail: float = None
    grad_norm: float = None
    cur_lr: float = None
    cur_wd: float = None
    cur_it: str = ''
    cur_ep: str = ''
    remain_time: str = ''
    finish_time: str = ''
    local_out_dir_path: str = 'results/' + exp_name
    log_txt_path: str = '...'
    last_ckpt_path: str = '...'

    tf32: bool = True
    device: str = 'cpu'         # [automatically set]
    seed: int = 3407
    same_seed_for_all_ranks: int = 0     # only for the distributed sampler

    def seed_everything(self, benchmark: bool):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = benchmark
        if self.seed is None:
            torch.backends.cudnn.deterministic = False
        else:
            torch.backends.cudnn.deterministic = True
            seed = self.seed * dist.get_world_size() + dist.get_rank()
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

    def get_different_generator_for_each_rank(self) -> Optional[torch.Generator]:   # for random augmentation
        if self.seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(self.seed * dist.get_world_size() + dist.get_rank())
        return g

    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        for k in self.class_variables.keys():
            if k not in {'device'}:     # not serializable
                d[k] = getattr(self, k)
        return d

    def load_state_dict(self, d: Union[OrderedDict, dict]):
        for k in d.keys():
            setattr(self, k, d[k])

    @staticmethod
    def set_tf32(tf32: bool):
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
                print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')

    def dump_log(self):
        if not dist.is_local_master():
            return
        if '1/' in self.cur_ep:     # first dump
            with open(self.log_txt_path, 'w') as fp:
                json.dump({'is_master': dist.is_master(), 'name': self.exp_name, 'cmd': self.cmd, 'commit': self.commit_id, 'branch': self.branch}, fp, indent=0)
                fp.write('\n')
        log_dict = {}
        for k, v in {
            'it': self.cur_it, 'ep': self.cur_ep,
            'lr': self.cur_lr, 'wd': self.cur_wd, 'grad_norm': self.grad_norm,
            'L_mean': self.L_mean, 'L_tail': self.L_tail, 'acc_mean': self.acc_mean, 'acc_tail': self.acc_tail,
            'vL_mean': self.vL_mean, 'vL_tail': self.vL_tail, 'vacc_mean': self.vacc_mean, 'vacc_tail': self.vacc_tail,
            'remain_time': self.remain_time, 'finish_time': self.finish_time,
        }.items():
            log_dict[k] = v.item() if hasattr(v, 'item') else v
        with open(self.log_txt_path, 'a') as fp:
            fp.write(f'{log_dict}\n')

    def __str__(self):
        s = '\n'.join(f'  {k:20s}: {getattr(self, k)}' for k in self.class_variables.keys() if k != 'device')
        return f'{{\n{s}\n}}\n'


def init_dist_and_get_args():
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('--local-rank=') or sys.argv[i].startswith('--local_rank='):
            del sys.argv[i]
            break
    args = Args(explicit_bool=True).parse_args(known_only=True)
    if len(args.extra_args) > 0:
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================\n{args.extra_args}')

    args.local_out_dir_path = 'results/' + args.exp_name
    from utils import misc
    os.makedirs(args.local_out_dir_path, exist_ok=True)
    misc.init_distributed_mode(local_out_path=args.local_out_dir_path, timeout=30)

    args.set_tf32(args.tf32)
    args.seed_everything(benchmark=True)

    args.device = dist.get_device()
    args.patch_nums = tuple(map(int, args.pn.replace('-', '_').split('_')))
    args.resos = tuple(pn * args.patch_size for pn in args.patch_nums)
    args.data_load_reso = max(args.resos)

    args.bs = args.glb_batch_size = args.batch_size * dist.get_world_size()
    args.workers = min(max(0, args.workers), 80)
    args.tlr = args.ac * args.tblr
    args.twde = args.twde or args.twd
    if args.wp == 0:
        args.wp = args.ep * 1 / 50

    args.log_txt_path = os.path.join(args.local_out_dir_path, 'log.txt')
    args.last_ckpt_path = os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth')
    return args
