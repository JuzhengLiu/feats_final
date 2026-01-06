import argparse
import os
import os.path as osp
import time
import random
import numpy as np
import torch
import torch.utils.data as tdata
import torch.nn.functional as F
from tqdm import tqdm

from torch.cuda.amp import GradScaler, autocast
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from utils.utils import update_args, Recorder, move_to, DATASET, MODEL, LOSS
import data.dataset as _data_reg
import utils.losses as _loss_reg
import models as _model_reg


HOME = osp.expanduser('/root/autodl-tmp/0-pipei-dinov3')

def load_feat(savedir: str, view: str):
    feat = torch.load(osp.join(savedir, f'{view}_feat'), map_location='cpu').to(torch.float32)
    gid = torch.load(osp.join(savedir, f'{view}_id'), map_location='cpu')
    name = torch.load(osp.join(savedir, f'{view}_name'))
    return feat, gid, name

def compute_mAP(index, good_index, junk_index):
    ap = 0.0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc
    mask = np.isin(index, junk_index, invert=True)
    index = index[mask]
    ngood = len(good_index)
    mask = np.isin(index, good_index)
    rows_good = np.argwhere(mask == True).flatten()
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2
    return ap, cmc

def eval_query(qf, ql, gf, gl):
    score = gf @ qf.unsqueeze(-1)
    score = score.squeeze().cpu().numpy()
    index = np.argsort(score)[::-1]
    query_index = np.argwhere(gl == ql)
    good_index = query_index
    junk_index = np.argwhere(gl == -1)
    ap, cmc = compute_mAP(index, good_index, junk_index)
    return ap, cmc

def evaluate_model(model, opt, device):
    """
    修改版：支持 SUES 多高度评估，修复了 Tensor 维度不匹配的 Bug
    """
    savedir = osp.join(HOME, opt.eval.feat)
    
    # SUES D2S 模式: Gallery=Sat, Query=Dro
    gall_feat, gid, gname = load_feat(savedir, 'sat')
    query_feat, qid, qname = load_feat(savedir, 'dro')
    
    model.eval()
    
    # --- Inference Logic ---
    batch_size = opt.eval.batch_size
    
    def get_inference_feat(raw_feats, is_query=False):
        outs = []
        for i in range(0, raw_feats.shape[0], batch_size):
            batch = raw_feats[i:i+batch_size].to(device)
            with torch.no_grad():
                if isinstance(model, torch.nn.DataParallel):
                    mod = model.module
                else:
                    mod = model

                if hasattr(mod, 'shared_enc') and hasattr(mod, 'moe_layer'):
                    base = mod.shared_enc(batch)
                    if is_query:
                        delta, _ = mod.moe_layer(base)
                        out = base + delta
                    else:
                        out = base
                elif hasattr(mod, 'enc1') and hasattr(mod, 'enc2'):
                    if not is_query: 
                        res = mod.enc1(batch)
                    else: 
                        res = mod.enc2(batch)
                    out = res[0] if isinstance(res, tuple) else res
                else:
                    out = batch

                out = F.normalize(out, dim=-1)
            outs.append(out.cpu())
        return torch.cat(outs, dim=0)

    # 计算特征
    g_enc = get_inference_feat(gall_feat, is_query=False)
    q_enc = get_inference_feat(query_feat, is_query=True)
        
    gl = gid.cpu().numpy()
    ql = qid.cpu().numpy()
    
    # --- Evaluation Logic ---
    
    # 1. Overall Evaluation (全量评估)
    CMC = torch.IntTensor(len(gid)).zero_()
    ap = 0.0
    for i in range(len(qid)):
        ap_tmp, CMC_tmp = eval_query(q_enc[i], ql[i], g_enc, gl)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    
    AP_all = ap / len(qid)
    CMC_all = CMC.float() / len(qid)
    
    top1p = len(CMC_all) // 100 if len(CMC_all) >= 100 else max(len(CMC_all) - 1, 0)
    print(f'Overall: top-1:{CMC_all[0]:.2%} | top-5:{CMC_all[4]:.2%} | top-1%:{CMC_all[top1p]:.2%} | AP:{AP_all:.2%}')

    # 2. Multi-Height Evaluation (分高度评估)
    heights = ['150', '200', '250', '300']
    
    for h in heights:
        # 筛选 Query
        q_idxs = [i for i, n in enumerate(qname) if str(n).startswith(h)]
        if len(q_idxs) == 0:
            continue
            
        q_sub = q_enc[q_idxs]
        ql_sub = ql[q_idxs]
        
        # 筛选 Gallery
        g_idxs = [i for i, n in enumerate(gname) if str(n).startswith(h)]
        if len(g_idxs) > 0:
            g_sub = g_enc[g_idxs]
            gl_sub = gl[g_idxs]
        else:
            g_sub = g_enc
            gl_sub = gl
            
        # 子集评估
        ap_h = 0.0
        # === [FIX] === 
        # 这里使用 len(gl_sub) 而不是 len(gl)，因为 CMC_tmp 的长度取决于当前子集 Gallery 的大小
        cmc_h = torch.IntTensor(len(gl_sub)).zero_() 
        valid_cnt = 0
        
        for i in range(len(ql_sub)):
            ap_tmp, CMC_tmp = eval_query(q_sub[i], ql_sub[i], g_sub, gl_sub)
            if CMC_tmp[0] != -1:
                cmc_h = cmc_h + CMC_tmp
                ap_h += ap_tmp
                valid_cnt += 1
        
        if valid_cnt > 0:
            ap_h /= valid_cnt
            cmc_h = cmc_h.float() / valid_cnt
            print(f'[{h}m]   : top-1:{cmc_h[0]:.2%} | AP:{ap_h:.2%}')

    return AP_all, CMC_all


def train(model, dataloader, criterions, optimizer, scheduler, scaler, recorder, device, opt):
    lambda_moe = getattr(opt.train, 'lambda_moe', 0.01)
    
    for epoch in range(opt.train.epochs):
        # 注入 current_epoch 供 Loss 模块使用
        opt.current_epoch = epoch
        
        model.train()
        optimizer.zero_grad(set_to_none=True)
        recorder.reset()
        current_lr = scheduler.get_last_lr() if scheduler is not None else [opt.train.lr]
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                    desc=f'Epoch {epoch+1}/{opt.train.epochs}')
        
        for batch_idx, data in pbar:
            start_t = time.time()
            if scaler:
                with autocast():
                    data = move_to(data, device)
                    data = model(data)
                    loss = criterions(data, model.logit_scale, opt)
                    
                    if 'aux_loss' in data:
                        moe_loss = data['aux_loss'] * lambda_moe
                        loss += moe_loss
                        recorder.update('L_moe', moe_loss.item(), opt.train.batch_size)
                    
                scaler.scale(loss).backward()
                if opt.train.clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_value_(model.parameters(), opt.train.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
            else:
                data = move_to(data, device)
                data = model(data)
                loss = criterions(data, model.logit_scale, opt)
                
                if 'aux_loss' in data:
                    moe_loss = data['aux_loss'] * lambda_moe
                    loss += moe_loss
                    recorder.update('L_moe', moe_loss.item(), opt.train.batch_size)

                loss.backward()
                if opt.train.clip_grad:
                    torch.nn.utils.clip_grad_value_(model.parameters(), opt.train.clip_grad)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
            
            recorder.update('Loss', loss.item(), opt.train.batch_size)
            recorder.update('FPS', opt.train.batch_size / (time.time() - start_t))
            
            postfix_dict = {
                'lr': f"{current_lr[0]:.6f}",
                'loss': f"{recorder.meter_dict['Loss']['meter'].avg:.4f}",
                'acc': f"{recorder.meter_dict['real_acc']['meter'].avg:.2%}",
            }
            if 'sk_eps' in recorder.meter_dict:
                postfix_dict['eps'] = f"{recorder.meter_dict['sk_eps']['meter'].avg:.3f}"
            
            pbar.set_postfix(postfix_dict)

        if ((epoch + 1) % opt.train.save_epoch == 0) or ((epoch + 1) == opt.train.epochs):
            state = {'model': model.state_dict(), 'epoch': epoch + 1}
            save_dir = osp.join(opt.result_path, opt.cfg.split('/')[-1].split('.')[0])
            os.makedirs(save_dir, exist_ok=True)
            torch.save(state, osp.join(save_dir, f'{epoch+1}_param.t'))
        
        if ((epoch + 1) % opt.train.eval_epoch == 0) or ((epoch + 1) == opt.train.epochs):
            print(f"\nEvaluating at epoch {epoch+1}...")
            AP, CMC = evaluate_model(model, opt, device)


def parse():
    parser = argparse.ArgumentParser(description='Cross-Modality Self-Supervised Training')
    parser.add_argument('cfg', type=str, help='yaml config path')
    parser.add_argument('--mixed_precision', default=True, type=bool)
    parser.add_argument('--gpus', '-g', default='0', type=str)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--result_path', default='outputs', type=str)
    parser.add_argument('--extend', action='store_true')
    parser.add_argument('--out_dim', type=int)
    parser.add_argument('--feat', type=str)
    parser.add_argument('--train_feat', type=str)
    parser.add_argument('--eval_feat', type=str)
    args = parser.parse_args()
    opt = update_args(args)
    if getattr(opt, 'out_dim', None) is not None:
        opt.model['out_dim'] = opt.out_dim
    if getattr(opt, 'feat', None) is not None:
        opt.train['feat'] = opt.feat
    if getattr(opt, 'train_feat', None) is not None:
        opt.train['feat'] = opt.train_feat
    if getattr(opt, 'eval_feat', None) is not None:
        opt.eval['feat'] = opt.eval_feat
    return opt


def main():
    opt = parse()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    gpu_str = str(opt.gpus).strip()
    device = 'cpu'
    if torch.cuda.is_available():
        if ',' in gpu_str or gpu_str == '':
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
            device = 'cuda:0'
        else:
            try:
                gpu_idx = int(gpu_str)
                device = f'cuda:{gpu_idx}'
            except Exception:
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
                device = 'cuda:0'

    os.makedirs(opt.result_path, exist_ok=True)

    trainset = DATASET[opt.train.dataset](**opt.train)
    trainloader = tdata.DataLoader(trainset, batch_size=opt.train.batch_size, shuffle=True, num_workers=opt.workers, drop_last=True, pin_memory=True)

    start_epoch = 0
    model = MODEL[opt.model.name](**opt.model)
    if opt.resume:
        load_param = opt.resume
        if osp.isfile(load_param):
            checkpoint = torch.load(load_param)
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)

    optimizer = model.build_opt(opt)

    if torch.cuda.device_count() > 1 and ',' in gpu_str:
        model = torch.nn.DataParallel(model)
    
    model.to(device)

    recorder = Recorder({"FPS": 'd', "Loss": 'f', "L_moe": 'f', "L_hard": 'f', "L_soft": 'f', "sk_eps": 'f'})
    criterions = LOSS[opt.train.loss[0]](opt.train.loss_w[0], opt, recorder, device)

    scaler = GradScaler(init_scale=2.**10) if opt.train.mixed_precision else None

    train_steps = len(trainloader) * opt.train.epochs
    warmup_steps = len(trainloader) * opt.train.warm_epochs
    if opt.train.scheduler == "polynomial":
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_training_steps=train_steps, lr_end=opt.train.lr_end, power=1.5, num_warmup_steps=warmup_steps)
    elif opt.train.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=train_steps, num_warmup_steps=warmup_steps)
    elif opt.train.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    else:
        scheduler = None

    train(model, trainloader, criterions, optimizer, scheduler, scaler, recorder, device, opt)


if __name__ == '__main__':
    main()