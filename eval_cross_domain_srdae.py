import argparse
import os
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import yaml
import sys

# 确保项目根目录在 sys.path 中
sys.path.append(os.getcwd())

from models.autoenc_moe import AutoEnc_MoE
from models.unsupervised_fusion import SRDAE

# ================= 默认配置区域 =================

#512
# # 1. University-1652 配置与权重 (Source Domain)
# U1652_CONFIG_PATH = 'configs/base_moe_shared_specific1.yml'
# # 请确保此路径指向你实际的 University 权重
# DEFAULT_U1652_WEIGHTS = 'outputs/base_moe_shared_specific1/300_param28dec.t'

# # 2. SUES-200 SRDAE 权重 (Target Domain Unsupervised)
# # 指向你在 SUES 上训练的 SRDAE 模型
# DEFAULT_SRDAE_PATH = '/root/autodl-tmp/0-pipei-dinov3/outputs/unsupervised_srdae_512/srdae_model_26_28.pth'

# # 3. SUES-200 特征路径
# FEAT_ROOT_D2S = '/root/autodl-tmp/0-pipei-dinov3/feats_sues_test/dinov3_vith16plus'
# FEAT_ROOT_S2D = '/root/autodl-tmp/0-pipei-dinov3/feats_sues_test_s2d/dinov3_vith16plus'
# # FEAT_ROOT_D2S = '/root/autodl-tmp/0-pipei-dinov3/feats_sues_test_256/dinov3_vith16plus'
# # FEAT_ROOT_S2D = '/root/autodl-tmp/0-pipei-dinov3/feats_sues_test_s2d_256/dinov3_vith16plus'

#256
# 1. University-1652 配置与权重 (Source Domain)
U1652_CONFIG_PATH = 'configs/base_moe_shared_specific.yml'
# 请确保此路径指向你实际的 University 权重
DEFAULT_U1652_WEIGHTS = 'outputs/base_moe_shared_specific/300_param.t'

# 2. SUES-200 SRDAE 权重 (Target Domain Unsupervised)
# 指向你在 SUES 上训练的 SRDAE 模型
DEFAULT_SRDAE_PATH = '/root/autodl-tmp/0-pipei-dinov3/outputs/unsupervised_srdae_256/srdae_model_26_28.pth'

# 3. SUES-200 特征路径
# FEAT_ROOT_D2S = '/root/autodl-tmp/0-pipei-dinov3/feats_sues_test/dinov3_vith16plus'
# FEAT_ROOT_S2D = '/root/autodl-tmp/0-pipei-dinov3/feats_sues_test_s2d/dinov3_vith16plus'
FEAT_ROOT_D2S = '/root/autodl-tmp/0-pipei-dinov3/feats_sues_test_256/dinov3_vith16plus'
FEAT_ROOT_S2D = '/root/autodl-tmp/0-pipei-dinov3/feats_sues_test_s2d_256/dinov3_vith16plus'

# 4. 融合参数
FUSION_LAYERS = [26, 28] # SRDAE 输入层
INPUT_DIM_B = 2560   # 1280 * 3
LATENT_DIM_B = 1280  # SRDAE 输出维度
LAYER_MAIN = 28      # MoE 输入层

# ===============================================

def load_feat(savedir, view):
    feat_path = osp.join(savedir, f'{view}_feat')
    id_path = osp.join(savedir, f'{view}_id')
    name_path = osp.join(savedir, f'{view}_name')
    
    if not osp.exists(feat_path):
        raise FileNotFoundError(f"Feature not found: {feat_path}")
        
    feat = torch.load(feat_path, map_location='cpu').float()
    gid = torch.load(id_path, map_location='cpu')
    name = torch.load(name_path)
    return feat, gid, name

def compute_mAP_standard(index, good_index, junk_index):
    ap = 0
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
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        ap += precision
    ap = ap / ngood
    return ap, cmc

def eval_query(qf, ql, gf, gl):
    score = gf @ qf.unsqueeze(-1)
    score = score.squeeze().cpu().numpy()
    index = np.argsort(score)[::-1]
    query_index = np.argwhere(gl == ql)
    good_index = query_index
    junk_index = np.argwhere(gl == -1)
    ap, cmc = compute_mAP_standard(index, good_index, junk_index)
    return ap, cmc

def main():
    parser = argparse.ArgumentParser(description='Cross-Domain Eval: U1652(MoE) + SUES(SRDAE)')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--mode', default='D2S', choices=['D2S', 'S2D'])
    parser.add_argument('--no_fusion', action='store_true', help='If set, only evaluate U1652 MoE model (Source Only).')
    parser.add_argument('--u1652_weights', default=DEFAULT_U1652_WEIGHTS, type=str)
    parser.add_argument('--srdae_path', default=DEFAULT_SRDAE_PATH, type=str)
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device} | Mode: {args.mode} | Fusion: {not args.no_fusion}")

    # 路径选择
    current_feat_root = FEAT_ROOT_S2D if args.mode == 'S2D' else FEAT_ROOT_D2S
    print(f"Feature Root: {current_feat_root}")

    # =========================================================
    # 1. Feature A: University-1652 MoE (Source Model)
    # =========================================================
    print(f"==> Loading Source Model (U1652) from {args.u1652_weights}...")
    with open(U1652_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    # 强制保留 University 的配置以匹配权重形状
    config['model']['query_num'] = 700 
    
    moe = AutoEnc_MoE(**config['model'])
    
    if not osp.exists(args.u1652_weights):
        raise FileNotFoundError(f"Source weights not found: {args.u1652_weights}")

    ckpt = torch.load(args.u1652_weights, map_location='cpu')
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    
    try:
        moe.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"[Error] Loading U1652 weights failed. Detail: {e}")
        return
        
    moe.to(device).eval()

    # 加载主层特征
    print(f"  -> Extracting Feature A (Layer {LAYER_MAIN})...")
    path_main = osp.join(current_feat_root, str(LAYER_MAIN))
    sat_feat_main, sat_id, sat_name = load_feat(path_main, 'sat')
    dro_feat_main, dro_id, dro_name = load_feat(path_main, 'dro')
    
    # MoE 推理函数
    batch_size = 256
    def get_moe_feat(feat, view_mode):
        outs = []
        for i in range(0, feat.shape[0], batch_size):
            batch = feat[i:i+batch_size].to(device)
            with torch.no_grad():
                base = moe.shared_enc(batch)
                if view_mode == 'dro':
                    delta, _ = moe.moe_layer(base)
                    out = base + delta
                else:
                    out = base
                out = F.normalize(out, dim=-1)
            outs.append(out.cpu())
        return torch.cat(outs, dim=0)

    # 提取 A 特征
    sat_A = get_moe_feat(sat_feat_main, 'sat')
    dro_A = get_moe_feat(dro_feat_main, 'dro')
    print(f"  -> Feature A Shape: {sat_A.shape}")

    # =========================================================
    # 2. Feature B: SUES SRDAE (Target Unsupervised)
    # =========================================================
    sat_B, dro_B = None, None
    use_fusion = not args.no_fusion

    if use_fusion:
        print(f"==> Loading Target Model (SRDAE) from {args.srdae_path}...")
        if not osp.exists(args.srdae_path):
            print(f"[Warning] SRDAE path invalid. Falling back to No Fusion.")
            use_fusion = False
        else:
            srdae = SRDAE(input_dim=INPUT_DIM_B, latent_dim=LATENT_DIM_B).to(device)
            srdae.load_state_dict(torch.load(args.srdae_path, map_location=device))
            srdae.eval()
            
            # 加载融合层
            sat_list, dro_list = [], []
            print(f"  -> Loading Fusion Layers: {FUSION_LAYERS}")
            for layer in FUSION_LAYERS:
                path = osp.join(current_feat_root, str(layer))
                s_f, _, _ = load_feat(path, 'sat')
                d_f, _, _ = load_feat(path, 'dro')
                sat_list.append(F.normalize(s_f, dim=-1))
                dro_list.append(F.normalize(d_f, dim=-1))
            
            sat_cat = torch.cat(sat_list, dim=-1)
            dro_cat = torch.cat(dro_list, dim=-1)
            
            # SRDAE 推理
            def get_srdae_feat(data):
                outs = []
                for i in range(0, data.shape[0], batch_size):
                    batch = data[i:i+batch_size].to(device)
                    with torch.no_grad():
                        z, _ = srdae(batch)
                        z = F.normalize(z, dim=-1)
                    outs.append(z.cpu())
                return torch.cat(outs, dim=0)

            sat_B = get_srdae_feat(sat_cat)
            dro_B = get_srdae_feat(dro_cat)
            print(f"  -> Feature B Shape: {sat_B.shape}")

    # =========================================================
    # 3. Fusion & Assign
    # =========================================================
    print("==> Fusing Features...")
    if use_fusion:
        feat_sat_final = torch.cat([sat_A, sat_B], dim=-1)
        feat_dro_final = torch.cat([dro_A, dro_B], dim=-1)
        # Final Norm
        feat_sat_final = F.normalize(feat_sat_final, dim=-1)
        feat_dro_final = F.normalize(feat_dro_final, dim=-1)
    else:
        feat_sat_final = sat_A
        feat_dro_final = dro_A

    print(f"  -> Final Dim: {feat_sat_final.shape[-1]}")

    # 分配 Query/Gallery
    if args.mode == 'D2S':
        # Query=Drone, Gallery=Sat
        q_feat, q_id, q_name = feat_dro_final, dro_id, dro_name
        g_feat, g_id, g_name = feat_sat_final, sat_id, sat_name
    else:
        # Query=Sat, Gallery=Drone
        q_feat, q_id, q_name = feat_sat_final, sat_id, sat_name
        g_feat, g_id, g_name = feat_dro_final, dro_id, dro_name
        
    g_feat = g_feat.to(device)

    # =========================================================
    # 4. Evaluation Loop
    # =========================================================
    print(f"\n--- Overall Evaluation ({args.mode}) ---")
    perform_eval("Overall", q_feat, q_id, g_feat, g_id, device)

    print(f"\n--- Multi-Height Evaluation ---")
    heights = ['150', '200', '250', '300']
    for h in heights:
        q_idxs = [i for i, n in enumerate(q_name) if str(n).startswith(h)]
        if len(q_idxs) == 0: continue
        
        q_sub = q_feat[q_idxs]
        qi_sub = q_id[q_idxs]
        
        # 筛选 Gallery (S2D模式下Drone是Gallery, 分高度; D2S下Sat是Gallery, 通常不分但可尝试)
        g_idxs = [i for i, n in enumerate(g_name) if str(n).startswith(h)]
        if len(g_idxs) > 0:
            g_sub = g_feat[g_idxs] # 已经是 tensor on gpu
            gi_sub = g_id[g_idxs]
        else:
            g_sub = g_feat
            gi_sub = g_id
            
        perform_eval(f"{h}m", q_sub, qi_sub, g_sub, gi_sub, device)

def perform_eval(tag, qf, qi, gf, gi, device):
    gl = gi.cpu().numpy()
    ql = qi.cpu().numpy()
    
    CMC = torch.IntTensor(len(gl)).zero_()
    ap = 0.0
    
    for i in tqdm(range(len(ql)), desc=f'Eval {tag}'):
        q_vec = qf[i].to(device)
        ap_tmp, CMC_tmp = eval_query(q_vec, ql[i], gf, gl)
        if CMC_tmp[0] != -1:
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            
    AP = ap / len(ql)
    CMC = CMC.float() / len(ql)
    print(f'[{tag}] R@1: {CMC[0]:.2%} | AP: {AP:.2%}')

if __name__ == '__main__':
    main()