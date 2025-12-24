# -*- coding: utf-8 -*-
from typing import Sequence, Union, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from pathlib import Path
import re

# --------- 工具 ----------
def _to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)

def _pick_expr_matrix(adata, genes: Sequence[str], expr_source: str = "X"):
    """
    选择表达来源:
      - "X"       : adata.X
      - "raw"     : adata.raw.X（若存在）
      - "layer:xx": adata.layers['xx']
    返回: (cells x genes) 的 dense 矩阵、与之对齐的基因名列表
    """
    genes = list(map(str, genes))
    var_all = np.array(adata.var_names, dtype=str)
    var_set = set(var_all)

    genes_in = [g for g in genes if g in var_set]
    if len(genes_in) == 0:
        raise ValueError("提供的 marker 基因一个也不在 adata.var_names 中。")

    if expr_source == "raw":
        if getattr(adata, "raw", None) is None:
            raise ValueError("选择了 expr_source='raw'，但 adata.raw 不存在。")
        raw_vars = np.array(adata.raw.var_names, dtype=str)
        raw_set = set(raw_vars)
        genes_in_raw = [g for g in genes_in if g in raw_set]
        if len(genes_in_raw) == 0:
            raise ValueError("这些基因不在 adata.raw 中。请改用 expr_source='X' 或指定 layer。")
        M = _to_dense(adata.raw[:, genes_in_raw].X)
        return M, genes_in_raw

    if expr_source.startswith("layer:"):
        layer = expr_source.split(":", 1)[1]
        if layer not in adata.layers:
            raise ValueError(f"在 adata.layers 中找不到 '{layer}'。")
        col_idx = np.where(np.isin(var_all, genes_in))[0]
        M = _to_dense(adata.layers[layer][:, col_idx])
        return M, list(var_all[col_idx])

    # 默认用 X
    col_idx = np.where(np.isin(var_all, genes_in))[0]
    M = _to_dense(adata.X[:, col_idx])
    return M, list(var_all[col_idx])

def _transform_expr(M: np.ndarray, how: str = "cpm_log1p") -> np.ndarray:
    """
    表达变换:
      - "none"       : 不变换
      - "log1p"      : log1p
      - "cpm_log1p"  : 每细胞 CPM(1e4) 后 log1p
    """
    if how == "none":
        return M
    if how == "log1p":
        return np.log1p(M)
    if how == "cpm_log1p":
        s = M.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        M_cpm = M / s * 1e4
        return np.log1p(M_cpm)
    raise ValueError(f"未知 expr_transform: {how}")

def _flatten_markers(markers: Union[Dict[str, Sequence[str]], pd.DataFrame, Sequence[str]],
                     cluster_key: Optional[str],
                     top_k: Optional[int]) -> List[str]:
    """
    将 {cluster: [genes]}、含 cluster/gene 列的 DataFrame，或直接的基因列表
    展平为去重基因列表（dict/DF 时按每簇 top_k 取）。
    """
    # 直接列表
    if isinstance(markers, (list, tuple, np.ndarray, pd.Index)):
        return list(map(str, markers))

    # dict: {cluster: [genes]}
    if isinstance(markers, dict):
        flat = []
        for _, gs in markers.items():
            if top_k is None:
                flat.extend(list(gs))
            else:
                flat.extend(list(gs)[:top_k])
        # 去重并保序
        seen, uniq = set(), []
        for g in flat:
            if g not in seen:
                seen.add(g); uniq.append(g)
        return uniq

    # DataFrame: 识别 cluster / gene 列
    if isinstance(markers, pd.DataFrame):
        df = markers.copy()
        cand_cluster = [cluster_key, "cluster", "group", "label", "domain", "PlantST_refine_domain"]
        ccol = next((c for c in cand_cluster if c and c in df.columns), None)
        cand_gene = ["gene", "genes", "name", "symbol", "feature"]
        gcol = next((c for c in cand_gene if c in df.columns), None)
        if ccol is None or gcol is None:
            raise ValueError("无法在 markers DataFrame 中识别 cluster/gene 列。")
        flat = []
        for _, sub in df.groupby(ccol):
            gs = list(map(str, sub[gcol].tolist()))
            if top_k is not None:
                gs = gs[:top_k]
            flat.extend(gs)
        seen, uniq = set(), []
        for g in flat:
            if g not in seen:
                seen.add(g); uniq.append(g)
        return uniq

    raise ValueError("markers 必须是 dict、DataFrame 或基因名序列。")

def _sanitize_filename(name: str) -> str:
    # 只保留安全字符，避免文件名非法
    name = re.sub(r"[^\w\-.]+", "_", str(name))
    return name.strip("._")

# --------- 主函数 ----------
def plot_marker_domain_trends_per_gene(
    adata,
    markers: Union[Dict[str, Sequence[str]], pd.DataFrame, Sequence[str]],
    cluster_key: str = "PlantST_refine_domain",
    save_dir: Union[str, Path] = "marker_domain_trends",
    expr_source: str = "X",
    expr_transform: str = "cpm_log1p",
    # 域序：默认用 pseudotime_signed 中位数排序（正=向内，负=向外）；
    # 若你已知道域的顺序，直接传 domain_order 覆盖。
    pseudotime_key: str = "pseudotime_signed",
    domain_order: Optional[Sequence[str]] = None,
    inward_positive: bool = True,   # True 表示正=向内（木质部）；False 表示正=向外
    show_points: bool = False,
    dpi: int = 180,
    figsize: tuple = (7, 4.5),
    top_k: Optional[int] = None     # 若 markers 为 dict/DF，则每簇取前 top_k；None 表示全取
) -> Tuple[Dict[str, str], List[str], pd.DataFrame]:
    """
    为“每个 marker 单独生成一张折线图”，并保存到 save_dir。
    横轴：由内→外的细胞域顺序；纵轴：该域内所有 spot 的该基因平均表达。

    返回:
      files_map: {gene -> 保存路径}
      order    : 采用的域顺序（由内→外）
      df_means : DataFrame(index=域, columns=基因名, 值=均值表达)
    """
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"在 adata.obs 中找不到 '{cluster_key}'。")

    # 1) 展平成基因列表
    genes = _flatten_markers(markers, cluster_key=cluster_key, top_k=top_k)
    if len(genes) == 0:
        raise ValueError("提供的 marker 列表为空。")

    # 2) 选择表达矩阵并变换
    M, genes_in = _pick_expr_matrix(adata, genes, expr_source=expr_source)
    M = _transform_expr(M, expr_transform)

    # 3) 确定域顺序
    clusters = adata.obs[cluster_key].astype(str).values
    uniq_domains = list(pd.unique(clusters))
    if domain_order is None:
        if pseudotime_key not in adata.obs.columns:
            raise KeyError(
                f"未提供 domain_order，且 adata.obs 中没有 '{pseudotime_key}' 用于排序。"
                "请传入 domain_order=['内侧域', ..., '外侧域']。"
            )
        pt = np.asarray(adata.obs[pseudotime_key].values, dtype=float)
        med = {d: np.median(pt[clusters == d]) for d in uniq_domains}
        # inward_positive=True: 正数大的是内侧，内→外 = 按中位数从大到小
        order = [k for k, v in sorted(med.items(), key=lambda kv: kv[1], reverse=inward_positive)]
    else:
        order = list(map(str, domain_order))

    # 4) 计算“域 × 基因”的均值矩阵
    df_means = pd.DataFrame(index=order, columns=genes_in, dtype=float)
    for d in order:
        mask = (clusters == d)
        if not np.any(mask):
            df_means.loc[d, :] = np.nan
            continue
        # 该域内所有 spot：对每个基因求均值
        block = M[mask, :]
        df_means.loc[d, :] = np.asarray(block.mean(axis=0)).ravel()

    # 5) 为每个基因单独出图
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    files_map: Dict[str, str] = {}

    x = np.arange(len(order))
    for g in genes_in:
        y = df_means[g].values.astype(float)
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.plot(x, y, label=g)
        if show_points:
            ax.scatter(x, y, s=22)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=0)
        ax.set_xlabel("Domains (inner->outer)")
        ax.set_ylabel("Mean expression (per-domain spots)")
        ax.set_title(f"Marker: {g}")
        # 单基因，无需图例
        fig.tight_layout()
        fname = f"{_sanitize_filename(g)}.png"
        out = str(save_dir / fname)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        files_map[g] = out

    return files_map, order, df_means

# -*- coding: utf-8 -*-
from typing import Sequence, Union, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path

# ---------- helpers ----------
def _to_dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)

def _flatten_markers(markers: Union[Dict[str, Sequence[str]], pd.DataFrame, Sequence[str]],
                     cluster_key: Optional[str] = None,
                     top_k: Optional[int] = None) -> List[str]:
    """支持 dict/DataFrame/list 三种输入，展平成不重复的基因列表。"""
    if isinstance(markers, (list, tuple, np.ndarray, pd.Index)):
        return list(map(str, markers))

    if isinstance(markers, dict):
        flat = []
        for _, gs in markers.items():
            gs = list(gs)
            flat.extend(gs if top_k is None else gs[:top_k])
        seen, uniq = set(), []
        for g in flat:
            if g not in seen:
                seen.add(g); uniq.append(g)
        return uniq

    if isinstance(markers, pd.DataFrame):
        df = markers.copy()
        cand_cluster = [cluster_key, "cluster", "group", "label", "domain", "PlantST_refine_domain"]
        ccol = next((c for c in cand_cluster if c and c in df.columns), None)
        cand_gene = ["gene", "genes", "name", "symbol", "feature"]
        gcol = next((c for c in cand_gene if c in df.columns), None)
        if ccol is None or gcol is None:
            raise ValueError("Cannot infer cluster/gene columns from markers DataFrame.")
        flat = []
        for _, sub in df.groupby(ccol):
            gs = list(map(str, sub[gcol].tolist()))
            flat.extend(gs if top_k is None else gs[:top_k])
        seen, uniq = set(), []
        for g in flat:
            if g not in seen:
                seen.add(g); uniq.append(g)
        return uniq

    raise ValueError("markers must be a dict, DataFrame, or a list of gene names.")

# ---------- main ----------
def make_marker_domain_means_table(
    adata,
    markers: Union[Dict[str, Sequence[str]], pd.DataFrame, Sequence[str]],
    cluster_key: str = "PlantST_refine_domain",
    save_path: Union[str, Path] = "marker_domain_means_raw.csv",
    # 原样使用 raw 计数；如想做 CPM/log，可把 expr_transform 改为 "cpm_log1p"/"log1p"
    expr_transform: str = "none",    # "none" | "cpm_log1p" | "log1p"
    top_k: Optional[int] = None,     # markers 为 dict/DF 时，每簇最多取前 top_k
    # 列顺序：不传则按域标签排序；若希望按“由内->外”，可传 domain_order 覆盖
    domain_order: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    从 adata.raw 取表达，生成 行=marker，列=域 的均值表达表并保存为 CSV。
    返回：(df, missing_genes)
    """
    if adata.raw is None:
        raise ValueError("adata.raw is None. Please ensure raw matrix exists or modify to use adata.X/layers.")

    if cluster_key not in adata.obs.columns:
        raise KeyError(f"`{cluster_key}` not found in adata.obs.")

    # 1) 准备基因列表与存在性检查
    genes = _flatten_markers(markers, cluster_key=cluster_key, top_k=top_k)
    if len(genes) == 0:
        raise ValueError("Empty marker list.")

    raw_vars = np.array(adata.raw.var_names, dtype=str)
    raw_set = set(raw_vars)
    genes_in = [g for g in genes if g in raw_set]
    missing = [g for g in genes if g not in raw_set]
    if len(genes_in) == 0:
        raise ValueError("None of the markers are found in adata.raw.var_names.")

    # 2) 取 raw 表达并可选变换
    M = adata.raw[:, genes_in].X
    M = _to_dense(M).astype("float64")

    if expr_transform == "cpm_log1p":
        s = M.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        M = np.log1p(M / s * 1e4)
    elif expr_transform == "log1p":
        M = np.log1p(M)
    elif expr_transform == "none":
        pass
    else:
        raise ValueError("expr_transform must be one of {'none','cpm_log1p','log1p'}.")

    # 3) 计算每个域的均值
    clusters = adata.obs[cluster_key].astype(str).values
    if domain_order is None:
        cols = sorted(pd.unique(clusters), key=lambda x: (str(x)))
    else:
        cols = list(map(str, domain_order))

    data = {}
    for d in cols:
        mask = (clusters == d)
        if not np.any(mask):
            data[d] = np.full(len(genes_in), np.nan, dtype=float)
        else:
            data[d] = M[mask, :].mean(axis=0)

    df = pd.DataFrame(data, index=genes_in)
    # 4) 保存
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, float_format="%.6f")  # 可按需改精度

    return df, missing
