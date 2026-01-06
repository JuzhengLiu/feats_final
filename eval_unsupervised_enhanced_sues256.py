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

# 引入新模型
from models.unsupervised_fusion import SRDAE
from models.autoenc_moe import AutoEnc_MoE 

# --- 默认配置与路径常量 (SUES-200 适配版) ---

# SR-DAE 模型路径 (需指向您训练好的 SUES SRDAE 模型)
DEFAULT_SRDAE_PATH = '/root/autodl-tmp/0-pipei-dinov3/outputs/sues_unsupervised_srdae_256/srdae_model_26_28.pth'
#srdae_model_23_26_29


# 特征路径配置 (SUES-200 测试集)
# D2S 模式下的测试集特征路径
FEAT_ROOT_D2S = '/root/autodl-tmp/0-pipei-dinov3/feats_sues_test_256/dinov3_vith16plus'
# S2D 模式下的测试集特征路径
FEAT_ROOT_S2D = '/root/autodl-tmp/0-pipei-dinov3/feats_sues_test_s2d_256/dinov3_vith16plus'

# 融合层配置 (SR-DAE 输入层，需与训练时保持一致)
#FUSION_LAYERS = [23, 26, 29]
FUSION_LAYERS = [26, 28]
INPUT_DIM_B = 2560   # 1280 * 3
LATENT_DIM_B = 1280  # SR-DAE 输出维度

# MoE 配置 (Feature A)
MOE_CONFIG_PATH = '/root/autodl-tmp/0-pipei-dinov3/configs_sues/sues_train_moe256.yml'
# 注意：这里需要指向您实际训练出的 MoE 权重文件，通常是 best_param.t 或指定 epoch 的权重
MOE_WEIGHT_PATH = '/root/autodl-tmp/0-pipei-dinov3/outputs/sues_train_moe256/300_param.t'
LAYER_MAIN = 28 # Feature A 的主层
#LAYER_MAIN = 29 # Feature A 的主层


def load_feat(savedir, view):
    """
    加载指定目录下的特征、ID 和 名称
    view: 'sat' 或 'dro'
    """
    feat_path = osp.join(savedir, f'{view}_feat')
    id_path = osp.join(savedir, f'{view}_id')
    name_path = osp.join(savedir, f'{view}_name')
    
    if not osp.exists(feat_path):
        raise FileNotFoundError(f"Feature not found: {feat_path}")
        
    feat = torch.load(feat_path, map_location='cpu').float()
    gid = torch.load(id_path, map_location='cpu')
    name = torch.load(name_path) # 加载文件名，用于区分高度
    return feat, gid, name

def compute_mAP_standard(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc

    # 移除 junk
    mask = np.isin(index, junk_index, invert=True)
    index = index[mask]

    # 找到 good 匹配
    ngood = len(good_index)
    mask = np.isin(index, good_index)
    rows_good = np.argwhere(mask == True).flatten()

    cmc[rows_good[0]:] = 1
    
    # 标准 AP 计算
    for i in range(ngood):
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        ap += precision
    
    ap = ap / ngood

    return ap, cmc

def eval_query(qf, ql, gf, gl):
    """
    单张 Query 对 Gallery 的评估
    """
    # 计算相似度 (已归一化，直接矩阵乘法)
    score = gf @ qf.unsqueeze(-1)
    score = score.squeeze().cpu().numpy()
    
    # 排序
    index = np.argsort(score)[::-1]
    
    # 获取 Ground Truth
    query_index = np.argwhere(gl == ql)
    good_index = query_index
    junk_index = np.argwhere(gl == -1)
    
    ap, cmc = compute_mAP_standard(index, good_index, junk_index)
    return ap, cmc

def main():
    parser = argparse.ArgumentParser(description='Eval SR-DAE (Unsupervised) + MoE for SUES-200')
    parser.add_argument('--gpu', default='0', type=str, help='gpu index')
    parser.add_argument('--srdae_path', default=DEFAULT_SRDAE_PATH, type=str, help='SR-DAE 模型路径')
    
    # 是否拼接 SR-DAE 向量
    parser.add_argument('--no_fusion', action='store_true', help='设置此参数则仅评估 Feature A (MoE)，不融合 SR-DAE。')
    
    # 评估模式切换 D2S / S2D
    parser.add_argument('--mode', default='D2S', choices=['D2S', 'S2D'], help='评估模式: D2S (Drone query) 或 S2D (Sat query)。')

    args = parser.parse_args()
    
    # === GPU 设置 ===
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
    
    # === 路径选择 ===
    if args.mode == 'S2D':
        current_feat_root = FEAT_ROOT_S2D
        print(f"==> Evaluation Mode: S2D (Satellite -> Drone)")
    else:
        current_feat_root = FEAT_ROOT_D2S
        print(f"==> Evaluation Mode: D2S (Drone -> Satellite)")
        
    print(f"==> Feature Root: {current_feat_root}")
    
    # === 是否启用融合 ===
    use_fusion = not args.no_fusion
    if use_fusion:
        print("==> Strategy: Fusion Enabled (MoE 2560 + SR-DAE 1280 = 3840)")
    else:
        print("==> Strategy: Fusion Disabled (MoE 2560 only)")

    # -------------------------------------------------------------
    # 1. Feature B: SR-DAE (仅当 use_fusion=True 时执行)
    # -------------------------------------------------------------
    sat_B = None
    dro_B = None
    sat_name = None # 用于高度筛选
    dro_name = None
    
    # 用于最终评估的 ID
    final_sat_id = None
    final_dro_id = None
    
    if use_fusion:
        print(f"==> Generating Feature B (SR-DAE Unsupervised) from {args.srdae_path}...")
        
        if not osp.exists(args.srdae_path):
            print(f"[Warning] SR-DAE Model not found at {args.srdae_path}. Proceeding might fail or require path correction.")

        # 初始化 SR-DAE
        srdae = SRDAE(input_dim=INPUT_DIM_B, latent_dim=LATENT_DIM_B).to(device)
        
        try:
            srdae.load_state_dict(torch.load(args.srdae_path, map_location=device))
        except Exception as e:
            print(f"[Error] Loading SRDAE failed: {e}")
            # 如果加载失败，可以考虑 return 或者继续只跑 MoE
            if not args.no_fusion: return

        srdae.eval()
        
        # 加载多层特征用于 SR-DAE 输入
        sat_list, dro_list = [], []
        
        print(f"  -> Loading Fusion Layers: {FUSION_LAYERS}")
        for layer in FUSION_LAYERS:
            path = osp.join(current_feat_root, str(layer))
            # 加载特征
            s_f, s_i, s_n = load_feat(path, 'sat')
            d_f, d_i, d_n = load_feat(path, 'dro')
            
            # 保存 ID 和 Name (只需要保存一次)
            if final_sat_id is None: 
                final_sat_id, final_dro_id = s_i, d_i
                sat_name, dro_name = s_n, d_n
            
            # Pre-Norm
            sat_list.append(F.normalize(s_f, dim=-1))
            dro_list.append(F.normalize(d_f, dim=-1))
            
        # 拼接多层特征
        sat_cat_b = torch.cat(sat_list, dim=-1)
        dro_cat_b = torch.cat(dro_list, dim=-1)
        
        # 批量推理由 SR-DAE 提取特征
        batch_size = 256
        def get_srdae_feat(data):
            outs = []
            for i in range(0, data.shape[0], batch_size):
                batch = data[i:i+batch_size].to(device)
                with torch.no_grad():
                    z, _ = srdae(batch) 
                    z = F.normalize(z, dim=-1) # Output Norm
                outs.append(z.cpu())
            return torch.cat(outs, dim=0)

        sat_B = get_srdae_feat(sat_cat_b)
        dro_B = get_srdae_feat(dro_cat_b)
        print(f"  -> Feature B (Sat/Dro) Shape: {sat_B.shape}")
    else:
        # 如果不融合，我们需要 ID 和 Name 来进行后续评估
        path_tmp = osp.join(current_feat_root, str(LAYER_MAIN))
        _, final_sat_id, sat_name = load_feat(path_tmp, 'sat')
        _, final_dro_id, dro_name = load_feat(path_tmp, 'dro')

    # -------------------------------------------------------------
    # 2. Feature A: MoE (始终执行)
    # -------------------------------------------------------------
    print("==> Generating Feature A (MoE)...")
    if not osp.exists(MOE_CONFIG_PATH):
        raise FileNotFoundError(f"MoE config not found at {MOE_CONFIG_PATH}")
        
    with open(MOE_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    # 强制覆盖维度配置以匹配 featsv14 架构
    config['model']['out_dim'] = 1280
    config['model']['vec_dim'] = 2560
    
    moe = AutoEnc_MoE(**config['model'])
    
    if osp.exists(MOE_WEIGHT_PATH):
        print(f"  -> Loading MoE weights from {MOE_WEIGHT_PATH}")
        checkpoint = torch.load(MOE_WEIGHT_PATH, map_location='cpu')
        moe.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    else:
        print(f"[Warning] MoE weights not found at {MOE_WEIGHT_PATH}. Using random init (Evaluation will be meaningless)!")
        
    moe.to(device).eval()
    
    # 加载主层特征 (Layer 28)
    path_main = osp.join(current_feat_root, str(LAYER_MAIN))
    sat_feat_main, _, _ = load_feat(path_main, 'sat')
    dro_feat_main, _, _ = load_feat(path_main, 'dro')

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

    # 提取 MoE 特征
    # 注意：在 SUES 代码逻辑中，通常 Drone 视为 Query (需要增强)，Sat 视为 Gallery
    # D2S: Query(Dro)->MoE, Gallery(Sat)->Base
    # S2D: Query(Sat)->Base, Gallery(Dro)->MoE (如果 Drone 也是作为 Gallery，也应增强)
    # 这里的 view_mode 传递 'dro' 或 'sat' 来决定是否过 MoE 层
    sat_A = get_moe_feat(sat_feat_main, 'sat')
    dro_A = get_moe_feat(dro_feat_main, 'dro')
    print(f"  -> Feature A (Sat/Dro) Shape: {sat_A.shape}")

    # -------------------------------------------------------------
    # 3. Concatenation & Mode Switching
    # -------------------------------------------------------------
    print(f"==> Preparing Final Features (Mode: {args.mode})...")

    # 3.1 融合
    if use_fusion:
        feat_sat_final = torch.cat([sat_A, sat_B], dim=-1)
        feat_sat_final = F.normalize(feat_sat_final, dim=-1)
        
        feat_dro_final = torch.cat([dro_A, dro_B], dim=-1)
        feat_dro_final = F.normalize(feat_dro_final, dim=-1)
    else:
        feat_sat_final = sat_A
        feat_dro_final = dro_A

    print(f"  -> Final Feature Dim: {feat_sat_final.shape[-1]}")

    # 3.2 分配 Query 和 Gallery
    if args.mode == 'D2S':
        # Query = Drone, Gallery = Satellite
        gallery_feat_all = feat_sat_final
        gallery_id_all = final_sat_id
        gallery_name_all = sat_name
        
        query_feat_all = feat_dro_final
        query_id_all = final_dro_id
        query_name_all = dro_name
    else:
        # S2D: Query = Satellite, Gallery = Drone
        gallery_feat_all = feat_dro_final
        gallery_id_all = final_dro_id
        gallery_name_all = dro_name
        
        query_feat_all = feat_sat_final
        query_id_all = final_sat_id
        query_name_all = sat_name

    # 将 Gallery 移至 GPU (如果显存允许)
    gallery_feat_all = gallery_feat_all.to(device)

    # -------------------------------------------------------------
    # 4. Evaluation Loop (Overall + Multi-Height)
    # -------------------------------------------------------------
    print("==> Starting SUES-200 Evaluation...")

    # 定义评估函数
    def perform_eval(tag, q_feat, q_id, g_feat, g_id):
        gl = g_id.cpu().numpy()
        ql = q_id.cpu().numpy()
        
        CMC = torch.IntTensor(len(gl)).zero_()
        ap = 0.0
        
        for i in tqdm(range(len(ql)), desc=f'Eval {tag}'):
            q_vec = q_feat[i].to(device)
            ap_tmp, CMC_tmp = eval_query(q_vec, ql[i], g_feat, gl)
            if CMC_tmp[0] != -1:
                CMC = CMC + CMC_tmp
                ap += ap_tmp
                
        AP = ap / len(ql)
        CMC = CMC.float() / len(ql)
        
        top1 = CMC[0]
        top5 = CMC[4] if len(CMC) > 4 else CMC[-1]
        
        print(f'[{tag}] R@1: {top1:.2%} | R@5: {top5:.2%} | AP: {AP:.2%}')
        return AP, top1

    # 4.1 Overall Evaluation
    print(f"\n--- Overall Evaluation ---")
    perform_eval("Overall", query_feat_all, query_id_all, gallery_feat_all, gallery_id_all)
    
    # 4.2 Multi-Height Evaluation
    heights = ['150', '200', '250', '300']
    print(f"\n--- Multi-Height Evaluation ---")
    
    for h in heights:
        # 筛选符合当前高度的 Query
        # 假设文件名格式包含 '150/', '200/' 等前缀
        q_idxs = [i for i, n in enumerate(query_name_all) if str(n).startswith(h)]
        
        if len(q_idxs) == 0:
            print(f"[{h}m] No query data found.")
            continue
            
        q_feat_sub = query_feat_all[q_idxs]
        q_id_sub = query_id_all[q_idxs]
        
        # 对于 Gallery，SUES 通常使用全量 Gallery (Cross-view)，或者对应高度的 Gallery
        # 如果是 D2S，Gallery 是卫星图。卫星图通常只有一份（或复制了多份）。
        # 如果 Gallery 也是按高度文件夹存储的 (prepare_sues.py 逻辑)，我们可以筛选对应高度的 Gallery
        g_idxs = [i for i, n in enumerate(gallery_name_all) if str(n).startswith(h)]
        
        if len(g_idxs) > 0:
            g_feat_sub = gallery_feat_all[g_idxs]
            g_id_sub = gallery_id_all[g_idxs]
        else:
            # 如果没有找到对应高度的 Gallery，则使用全量 Gallery
            g_feat_sub = gallery_feat_all
            g_id_sub = gallery_id_all
            
        perform_eval(f"{h}m", q_feat_sub, q_id_sub, g_feat_sub, g_id_sub)

    print(f'==================================================')
    print(f'Evaluation Config:')
    print(f'  Mode:   {args.mode}')
    print(f'  Fusion: {use_fusion}')
    print(f'==================================================')

if __name__ == '__main__':
    main()