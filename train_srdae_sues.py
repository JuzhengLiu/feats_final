import argparse
import os
import os.path as osp
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from collections import defaultdict

# 确保引入的是 featsv14 库中的工具
from utils.utils import update_args, mkdir_if_missing
from utils.c_adamw import CAdamW
from models.unsupervised_fusion import SRDAE
from utils.unsupervised_losses import UnsupervisedLoss

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_feat_layer(root, layer_idx, mode):
    # 路径逻辑: root/layer_idx/mode_feat
    path = osp.join(root, str(layer_idx))
    feat_path = osp.join(path, f'{mode}_feat')
    if not osp.exists(feat_path):
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    return torch.load(feat_path, map_location='cpu').float()

def prepare_training_data(feat_root, layers):
    print(f"==> Loading features from {feat_root}...")
    sat_feats_list = []
    dro_feats_list = []
    
    for layer in layers:
        s_f = load_feat_layer(feat_root, layer, 'sat')
        d_f = load_feat_layer(feat_root, layer, 'dro')
        
        # 归一化后加入列表
        sat_feats_list.append(F.normalize(s_f, dim=-1))
        dro_feats_list.append(F.normalize(d_f, dim=-1))
    
    # 拼接逻辑: [Sat_all, Dro_all]
    # cat(sat_feats_list, dim=-1) -> [N_sat, total_dim]
    # cat(dro_feats_list, dim=-1) -> [N_dro, total_dim]
    # cat([sat, dro], dim=0) -> [N_total, total_dim]
    train_X = torch.cat([torch.cat(sat_feats_list, dim=-1), 
                         torch.cat(dro_feats_list, dim=-1)], dim=0)
    print(f"==> Unsupervised Data: {train_X.shape}")
    return train_X

def train(model, dataloader, criterion, optimizer, scheduler, device, opt):
    print("==> Start SR-DAE Training...")
    
    for epoch in range(opt.train.epochs):
        model.train()
        loss_avg = defaultdict(float)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch_x in pbar:
            batch_x = batch_x[0].to(device)
            optimizer.zero_grad()
            
            z, recon = model(batch_x)
            loss, loss_dict = criterion(batch_x, recon, z, model)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            for k, v in loss_dict.items():
                loss_avg[k] += v.item()
            
            # 显示关键指标
            pbar.set_postfix({
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
                'Rec': f"{loss_dict['recon']:.4f}", 
                'Cov': f"{loss_dict['decov']:.4f}",
                'Spa': f"{loss_dict['sparse']:.4f}"
            })
            
        msg = f"Epoch {epoch+1}: "
        for k, v in loss_avg.items():
            msg += f"{k}: {v/len(dataloader):.4f} | "
        print(msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str)
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()
    opt = update_args(args)
    set_seed(42)

    gpu_str = str(args.gpu).strip()
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
    device = torch.device(device)
    print(f"Running on {device}")

    mkdir_if_missing(opt.output_dir)
    
    # 准备数据
    train_X = prepare_training_data(opt.feat_root, opt.fusion_layers)
    if train_X.shape[1] != opt.input_dim: 
        print(f"[Info] Adjusting input_dim from {opt.input_dim} to {train_X.shape[1]}")
        opt.input_dim = train_X.shape[1]
    
    # 构建 DataLoader
    dataset = TensorDataset(train_X)
    dataloader = DataLoader(dataset, batch_size=opt.train.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # 构建模型
    model = SRDAE(input_dim=opt.input_dim, latent_dim=opt.latent_dim, drop_rate=opt.train.drop_rate).to(device)
    optimizer = CAdamW(model.parameters(), lr=opt.train.lr)
    
    train_steps = len(dataloader) * opt.train.epochs
    warmup_steps = int(len(dataloader) * opt.train.warmup_epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, train_steps)
    
    # 构建 Loss
    lambda_sparse = getattr(opt.train, 'lambda_sparse', 0.0)
    
    criterion = UnsupervisedLoss(
        lambda_recon=opt.train.lambda_recon,
        lambda_decov=opt.train.lambda_decov,
        lambda_orth=opt.train.lambda_orth,
        lambda_sparse=lambda_sparse
    ).to(device)
    
    # 开始训练
    train(model, dataloader, criterion, optimizer, scheduler, device, opt)
    
    # 保存结果
    save_path = osp.join(opt.output_dir, opt.save_name)
    torch.save(model.state_dict(), save_path)
    print(f"==> Saved to {save_path}")

if __name__ == '__main__':
    main()