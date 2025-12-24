import os
import numpy as np
from numba.core.typing.builtins import Print
import pandas as pd
from PlantST import run
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from position import *
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.sparse import csr_matrix
import h5py
import torch.backends.cudnn as cudnn
import random
import SpatialDE
from two_time import *
from find_marker import *
from plot import *
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
data_path = "../data"              # to your path
data_name = 'outs_4'               # project name
save_path = "../Results/heng_4"    # save path
n_domains = 8
seed = 36 # 36
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 保证 CuDNN 算法确定性（如果在 GPU 上用到 CuDNN）
cudnn.deterministic = True
cudnn.benchmark = False
torch.use_deterministic_algorithms(True)  # 确保所有算法可复现

deepen = run(save_path=save_path,
             pre_epochs=1000,
             epochs=1000,
             use_gpu=True)

# Load the saved adata with clusters from domain identification
adata = sc.read_h5ad(f"{save_path}/{data_name}_with_clusters.h5ad")

graph_dict = deepen.get_graph(
    adata,
    adata.obsm["spatial"],
    distType="KDTree",
    k=4, rad_cutoff=50,
    n_components=50,
    enh=True
)
# 预处理
data_sf = deepen.data_preprocess_pseudotime(adata=adata, n_top_genes=2000)
# 训练生成表达嵌入+空间正则
adata.obsm["pseudotime_embed"] = deepen.feature_extration(adata=adata,
                                                          data=data_sf,
                                                          graph_dict=graph_dict,
                                                          task="pseudotime")
# 拟时序分析
pt = TwoBranchPseudotime(adata,
                         spatial_key="spatial",
                         embed_key="pseudotime_embed",
                         cluster_key="cluster",
                         cambium_label=4)

domain_order = ["1","3","7","4","5","2","0","6"]

comp_labels, info = pt.compute_two_branch_multi(
    domain_order=domain_order,
    n_neighbors=4,
    inward_positive=True,
    save_keys=("pseudotime_in", "pseudotime_out", "pseudotime_signed"),
    spatial_smooth=True,
    smooth_radius_factor=1.5,  # 空间平滑半径因子
    min_cluster_size=5,        # 最小聚类大小
    groupby=None,                     # None: 自动检测空间连通分量
    verbose=True                      # 打印进度信息
)
# 生成并保存图像
pt.plot_pseudotime_map(
    save_path=save_path,
    file_name="spatial_pseudotime",
    file_format="png",
    dpi=600,
    values="pseudotime_signed",
    colormap="vik",
    vmin=-1, vmax=1,
    show=True,
)
# 保存拟时序结果
pseudotime_keys = ["pseudotime_in", "pseudotime_out", "pseudotime_signed"]

# 确保类型正确（可选）
for k in pseudotime_keys:
    adata.obs[k] = adata.obs[k].astype(float)

# 存 h5ad（带 cluster 和 pSM）
adata.write_h5ad(f"{save_path}/{data_name}_with_clusters_pseudotime.h5ad",
                 compression="gzip")

# 只导出观测级信息到 csv，方便后续单独读
cols_to_save = ["cluster"] + pseudotime_keys
adata.obs[cols_to_save].to_csv(
    f"{save_path}/{data_name}_clusters_pseudotime.csv"
)

print("Saved:", f"{save_path}/{data_name}_with_clusters_pseudotime.h5ad")
print("Saved:", f"{save_path}/{data_name}_clusters_pseudotime.csv")

