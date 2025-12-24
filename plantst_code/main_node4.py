import os
# from statsmodels.sandbox.distributions.genpareto import shape
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
from scipy.sparse import csr_matrix, coo_matrix
import h5py
import torch.backends.cudnn as cudnn
import random
import SpatialDE
from two_time import *
from find_marker import *
from plot import *
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# -------------------- 路径与基础配置 --------------------
data_path = "../data"              # to your path
data_name = 'outs_4'               # project name
save_path = "../Results/heng_4"    # save path
n_domains = 8                     # number of spatial domains.

best_ari = -1
best_params = None
iteration = 0
max_iterations = 100

seed = 36
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 保证 CuDNN 确定性
cudnn.deterministic = True
cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# -------------------- PlantST 流程 --------------------
deepen = run(save_path=save_path, pre_epochs=800, epochs=1000, use_gpu=True)

# 读取数据
adata = deepen.get_adata(platform="Visium", data_path=data_path, data_name=data_name)

# 图像矩阵
adata = deepen.get_image_crop(adata, data_name=data_name)

# 增强矩阵生成
adata = deepen.get_augment(
    adata,
    spatial_type="LinearRegress",
    adjacent_weight=0.4,
    neighbour_k=6,
    spatial_k=4
)

# scale 标准化和 pca
data = deepen.data_preprocess_identify(adata, pca_n_comps=200)

# 邻接矩阵生成
graph_dict = deepen.get_graph(
    adata,
    adata.obsm["spatial"],
    distType="KDTree",
    k=4, rad_cutoff=100,
    n_components=100,
    enh=True
)

# 表达嵌入
adata.obsm["identify_embed"] = deepen.feature_extration(
    adata=adata,
    data=data,
    graph_dict=graph_dict,
    contrastive_weight=1,
    temperature=1,
    kl_weight=1,
    mse_weight=10,
    bce_kld_weight=0.1,
    k_pos=4, k_neg=4,
    update_interval=100,
    gene_sim_threshold_pos=0.7,
    gene_sim_threshold_neg=0.3,
    task="identify"
)

# 聚类
adata = deepen.get_cluster_data(
    adata,
    n_domains=n_domains,
    use_rep='identify_embed',
    priori=True
)

deepen.evaluate_clustering(adata.obsm["identify_embed"], adata.obs["cluster"].values, true_labels=None)

# -------------------- 空间图 & 保存 --------------------
sc.settings.figdir = save_path

sc.pl.spatial(
    adata,
    color="cluster",
    title="",
    frameon=False,
    spot_size=300,
    legend_loc=None,
    show=False
)
plt.tight_layout()
plt.savefig(f"{save_path}/identify_cluster_None.png", dpi=600, bbox_inches='tight')
plt.show()

adata.obs["cluster"] = adata.obs["cluster"].astype(str).astype("category")
adata.write_h5ad(f"{save_path}/{data_name}_with_clusters.h5ad", compression="gzip")
adata.obs[["cluster"]].to_csv(f"{save_path}/{data_name}_clusters.csv")
print("Saved:", f"{save_path}/{data_name}_with_clusters.h5ad")
# -------------------- UMAP 可视化（两套） --------------------
seed = 36
n_neighbors = 15
min_dist = 0.4
metric_embed = "cosine"
metric_data  = "euclidean"

# 1) identify_embed 的邻居与 UMAP
sc.pp.neighbors(
    adata,
    use_rep="identify_embed",
    n_neighbors=n_neighbors,
    metric=metric_embed,
    key_added="neighbors_identify"
)
sc.tl.umap(adata, neighbors_key="neighbors_identify", random_state=seed)
adata.obsm["umap_identify"] = adata.obsm["X_umap"].copy()

# 2) data 输入的邻居与 UMAP
adata.obsm["data_input"] = data if isinstance(data, np.ndarray) else data.toarray()
sc.pp.neighbors(
    adata,
    use_rep="data_input",
    n_neighbors=n_neighbors,
    metric=metric_data,
    key_added="neighbors_data"
 )
sc.tl.umap(adata, neighbors_key="neighbors_data", random_state=seed)
adata.obsm["umap_data"] = adata.obsm["X_umap"].copy()

# 绘制两张 UMAP（按 cluster 上色）
adata.obs["cluster"] = adata.obs["cluster"].astype("category")
sc.pl.embedding(adata, basis="umap_identify", color="cluster", legend_loc="none",
                frameon=False, title="", size=50, show=False)
plt.tight_layout(); plt.savefig(f"{save_path}/umap_identify.png", dpi=600, bbox_inches="tight"); plt.show()

sc.pl.embedding(adata, basis="umap_data", color="cluster", legend_loc="none",
                frameon=False, title="", size=50, show=False)
plt.tight_layout(); plt.savefig(f"{save_path}/umap_data.png", dpi=600, bbox_inches="tight"); plt.show()

# -------------------- 兼容旧版的“默认邻居”切换工具 --------------------
def _backup_neighbors(adata):
    return (
        adata.uns.get('neighbors', None),
        adata.obsp.get('connectivities', None),
        adata.obsp.get('distances', None)
    )

def _use_neighbors_key_as_default(adata, neighbors_key):
    """把指定 neighbors_key 的邻接设为默认 neighbors（老版函数只认默认）。"""
    adata.uns['neighbors'] = adata.uns[f'{neighbors_key}']
    adata.obsp['connectivities'] = adata.obsp[f'{neighbors_key}_connectivities']
    adata.obsp['distances'] = adata.obsp[f'{neighbors_key}_distances']

def _restore_neighbors(adata, bak):
    n_uns, n_conn, n_dist = bak
    if n_uns is not None: adata.uns['neighbors'] = n_uns
    if n_conn is not None: adata.obsp['connectivities'] = n_conn
    if n_dist is not None: adata.obsp['distances'] = n_dist

# -------------------- 计算 PAGA（兼容旧版） --------------------
# -------------------- 计算 PAGA（兼容旧版） --------------------
def run_paga_with_fallback(adata, groups, neighbors_key):
    """优先用新接口；不支持时，临时切换默认邻居再跑 PAGA。"""
    try:
        sc.tl.paga(adata, groups=groups, neighbors_key=neighbors_key)
    except TypeError:
        bak = _backup_neighbors(adata)
        _use_neighbors_key_as_default(adata, neighbors_key)
        sc.tl.paga(adata, groups=groups)
        _restore_neighbors(adata, bak)

# 方向对齐，避免 *paga 与原始 UMAP 出现镜像/旋转差异
def procrustes_align(X, Y):
    X0, Y0 = X - X.mean(0), Y - Y.mean(0)
    U, s, Vt = np.linalg.svd(X0.T @ Y0)
    R = U @ Vt
    X1 = X0 @ R
    scale = np.sqrt((Y0**2).sum()) / max(np.sqrt((X1**2).sum()), 1e-12)
    X1 = X1 * scale + Y.mean(0)
    return X1

def _get_groups_order(adata, groupby):
    """
    返回与 PAGA 连通矩阵一致的簇顺序。
    兼容几种老版本情况：
      - adata.uns['paga']['groups'] 是列名字符串（如 'cluster'）
      - 是分类对象（Categorical）
      - 是数组/列表
    """
    paga_uns = adata.uns.get("paga", {})
    g = paga_uns.get("groups", None)

    # 情况 1：是列名字符串
    if isinstance(g, str):
        if g in adata.obs:
            col = adata.obs[g]
            if hasattr(col, "cat"):
                return list(col.cat.categories)
            else:
                return sorted(pd.unique(col.astype(str)))
        # 兜底：用传入的 groupby 列
        col = adata.obs[groupby]
        return list(col.cat.categories) if hasattr(col, "cat") else sorted(pd.unique(col.astype(str)))

    # 情况 2：是 Categorical / pandas 对象
    try:
        if hasattr(g, "cat"):
            return list(g.cat.categories)
    except Exception:
        pass

    # 情况 3：数组/列表
    try:
        return list(pd.Index(g))
    except Exception:
        # 最后兜底：直接用 groupby 列顺序
        col = adata.obs[groupby]
        return list(col.cat.categories) if hasattr(col, "cat") else sorted(pd.unique(col.astype(str)))

# -------------------- 手动 overlay：彩色实心圈 + 黑色连线 --------------------
def manual_paga_overlay(adata, umap_key, groupby, out_png, threshold=0.03, node_size=160):
    X_bak = adata.obsm["X_umap"].copy()
    adata.obsm["X_umap"] = adata.obsm[umap_key].copy()

    fig, ax = plt.subplots(figsize=(6, 5))
    sc.pl.embedding(adata, basis="umap", color=groupby,
                    legend_loc="none", frameon=False, size=50,
                    ax=ax, show=False, title="")

    coords = pd.DataFrame(adata.obsm["X_umap"], index=adata.obs_names, columns=["x","y"])
    cats = _get_groups_order(adata, groupby)

    # 质心
    centroids = {}
    for c in cats:
        idx = (adata.obs[groupby] == c).values
        centroids[c] = coords.loc[idx, ["x","y"]].mean().values

    # 连通矩阵（按权重画黑线）
    conn = adata.uns["paga"]["connectivities"].tocoo()
    wmax = conn.data.max() if conn.data.size else 1.0
    for i, j, w in zip(conn.row, conn.col, conn.data):
        if i >= j or w < threshold:
            continue
        (xi, yi) = centroids[cats[i]]
        (xj, yj) = centroids[cats[j]]
        lw = 0.6 + 2.6 * (float(w) / wmax)
        ax.plot([xi, xj], [yi, yj], color="black", linewidth=lw, alpha=0.9, zorder=1)

    # 实心彩色圈 + 黑描边
    palette = adata.uns.get(f"{groupby}_colors", sc.pl.palettes.default_20)
    for k, c in enumerate(cats):
        (x, y) = centroids[c]
        ax.scatter([x], [y], s=node_size,
                   c=[palette[k % len(palette)]],
                   edgecolors="black", linewidths=1.4,
                   marker="o", zorder=3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    adata.obsm["X_umap"] = X_bak

# -------------------- PAGA：基于 identify_embed 的邻居 --------------------
groupby = "cluster"
adata.obs[groupby] = adata.obs[groupby].astype("category")
assert adata.obs[groupby].nunique() >= 2, "PAGA 需要至少两个簇！"

# 计算 PAGA（不再另存单独的 PAGA 图）
run_paga_with_fallback(adata, groups=groupby, neighbors_key="neighbors_identify")
sc.pl.paga(adata, threshold=0.0, frameon=False, show=False)
# 用 PAGA 初始化 UMAP，并对齐到原始 identify UMAP 的方向
sc.tl.umap(adata, neighbors_key="neighbors_identify",
           init_pos="paga", min_dist=min_dist, random_state=seed)
adata.obsm["umap_identify_paga"] = procrustes_align(
    adata.obsm["X_umap"].copy(), adata.obsm["umap_identify"]
)

# 只保存叠加图
manual_paga_overlay(
    adata, "umap_identify_paga", groupby,
    f"{save_path}/umap_identify_paga_overlay.png", threshold=0.03
)

# -------------------- PAGA：基于 data_input 的邻居（同样处理） --------------------
run_paga_with_fallback(adata, groups=groupby, neighbors_key="neighbors_data")
sc.pl.paga(adata, threshold=0.0, frameon=False, show=False)
sc.tl.umap(adata, neighbors_key="neighbors_data",
           init_pos="paga", min_dist=min_dist, random_state=seed)
adata.obsm["umap_data_paga"] = procrustes_align(
    adata.obsm["X_umap"].copy(), adata.obsm["umap_data"]
)

manual_paga_overlay(
    adata, "umap_data_paga", groupby,
    f"{save_path}/umap_data_paga_overlay.png", threshold=0.03
)
sc.pl.embedding(adata, basis="umap_data_paga", color=groupby,
                legend_loc="none", frameon=False, size=50, show=False, title="")
plt.tight_layout(); plt.savefig(f"{save_path}/umap_data_paga.png", dpi=600, bbox_inches="tight"); plt.show()

# # -------------------- Two-Branch 拟时序（保持你的原实现） --------------------
# data_sf = deepen.data_preprocess_pseudotime(adata=adata, n_top_genes=4000)
# adata.obsm["pseudotime_embed"] = deepen.feature_extration(
#     adata=adata,
#     data=data_sf,
#     graph_dict=graph_dict,
#     task="pseudotime"
# )
#
# pt = TwoBranchPseudotime(
#     adata,
#     spatial_key="spatial",
#     embed_key="pseudotime_embed",
#     cluster_key="cluster",
#     cambium_label="4"
# )
#
# domain_order = ["1","3","7","4","5","2","0","6"]
# comp_labels, info = pt.compute_two_branch_multi(
#     domain_order=domain_order,
#     n_neighbors=5,
#     inward_positive=True,
#     save_keys=("pseudotime_in", "pseudotime_out", "pseudotime_signed"),
#     spatial_smooth=True,
#     smooth_radius_factor=1.5,
#     min_cluster_size=5,
#     groupby=None,
#     verbose=True
# )
#
# print("\n=== 拟时序计算完成 ===")
# print(f"检测到 {len(info)} 个空间区域")
# print(f"拟时序范围: [{adata.obs['pseudotime_signed'].min():.3f}, {adata.obs['pseudotime_signed'].max():.3f}]")
#
# pt.plot_pseudotime_map(
#     save_path=save_path,
#     file_name="spatial_pseudotime",
#     file_format="png",
#     dpi=600,
#     values="pseudotime_signed",
#     colormap="vik",
#     vmin=-1, vmax=1,
#     show=True,
# )
# # 保存拟时序结果
# pseudotime_keys = ["pseudotime_in", "pseudotime_out", "pseudotime_signed"]
#
# # 确保类型正确（可选）
# for k in pseudotime_keys:
#     adata.obs[k] = adata.obs[k].astype(float)
#
# # 存 h5ad（带 cluster 和 pSM）
# adata.write_h5ad(f"{save_path}/{data_name}_with_clusters_pseudotime.h5ad",
#                  compression="gzip")
#
# # 只导出观测级信息到 csv，方便后续单独读
# cols_to_save = ["cluster"] + pseudotime_keys
# adata.obs[cols_to_save].to_csv(
#     f"{save_path}/{data_name}_clusters_pseudotime.csv"
# )
#
# print("Saved:", f"{save_path}/{data_name}_with_clusters_pseudotime.h5ad")
# print("Saved:", f"{save_path}/{data_name}_clusters_pseudotime.csv")

# # marker挖掘+可视化
# # 1) 实例化（设置全局默认）
# mf = MarkerFinder(
#     cluster_key="cluster",
#     spatial_key="spatial",
#     expr_source="raw",  # 若 adata.raw 不存在，改用 'X' 或某个 layer 名
#     expr_transform="cpm_log1p",
#     show_progress=True
# )
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# adata.raw = adata
# # # 2) 只用 DE（稳健 baseline）
# de_df, de_markers = mf.find(
#     adata, mode="de",
#     de_method="wilcoxon",
#     de_min_pct_in=0.3, de_max_pct_out=0.05,
#     de_min_logfc=0.05, # logFC 对数化的表达量倍数变化 越大 基因特异性越高
#     de_max_fdr=0.05, # 统计显著性
#     top_k=5
# )
#
# saved_files = mf.plot_markers(
#     adata, de_df,
#     cluster_key="cluster",
#     top_k=5,
#     output_dir=os.path.join(save_path, "marker_maps_DE"),
#     expr_source="raw", expr_transform="cpm_log1p",
#     img_key="hires", library_id=list(adata.uns["spatial"].keys())[0],
#     spot_size=300, alpha_img=1,
#     cmap="Reds",  # 低->浅，高->红
#     robust_p=99.5,
#     only_show_cluster=False  # 全局可视化（不是只显示该簇）
# )
#
# files_map, order, df_means = plot_marker_domain_trends_per_gene(
#     adata=adata,
#     markers=de_markers,  # 也可换成 de_df（DataFrame）或直接的基因名列表
#     cluster_key="cluster",
#     save_dir=os.path.join(save_path, "marker_domain_trends"),
#     expr_source="X",
#     expr_transform="cpm_log1p",
#     pseudotime_key="pseudotime_signed",  # 仅用于确定由内→外的域序
#     domain_order=None,  # 如果你已知道固定顺序，可传列表覆盖
#     inward_positive=True,  # 你的约定: 正=向内（木质部）
#     show_points=False,
#     dpi=600,
#     figsize=(7, 4.5),
#     top_k=5  # 每簇最多取 5 个 marker；用全部则设 None
# )
#
# print("每个基因输出文件：", files_map)
# print("域序（由内→外）：", order)
#
# # 用你已经筛出的 markers（dict），每簇取前 5 个；表达用 raw 原值
# df_mean, missing = make_marker_domain_means_table(
#     adata=adata,
#     markers=de_markers,  # 也可以传 de_df 或 直接的基因名列表
#     cluster_key="cluster",
#     save_path="marker_domain_means_raw.csv",
#     expr_transform="cpm_log1p",  # 保持 raw；若想用 CPM+log1p：改成 "cpm_log1p"
#     top_k=5,
#     domain_order=None  # 若你有固定的“内->外”顺序，传列表覆盖
# )



