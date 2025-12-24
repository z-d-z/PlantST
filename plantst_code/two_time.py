import os
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from scipy.spatial import cKDTree
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import anndata as ad

from scipy.spatial import ConvexHull


class TwoBranchPseudotime:
    def __init__(
            self,
            adata: ad.AnnData,
            *,
            spatial_key: str = "spatial",
            embed_key: str = "pseudotime_embed",
            cluster_key: str = "cluster",
            cambium_label: Any = 7,
    ):
        """
        参数:
            adata: AnnData对象
            spatial_key: 空间坐标在obsm中的键
            embed_key: 降维嵌入在obsm中的键（用于DPT计算）
            cluster_key: 聚类标签在obs中的键
            cambium_label: 形成层/中间层的标签
        """
        self.adata = adata
        self.spatial_key = spatial_key
        self.embed_key = embed_key
        self.cluster_key = cluster_key
        self.cambium_label = cambium_label
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """验证必需的数据键是否存在"""
        required = {
            f"obsm['{self.spatial_key}']": self.spatial_key in self.adata.obsm_keys(),
            f"obsm['{self.embed_key}']": self.embed_key in self.adata.obsm_keys(),
            f"obs['{self.cluster_key}']": self.cluster_key in self.adata.obs_keys(),
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise KeyError(f"Missing required keys: {', '.join(missing)}")

    def compute_two_branch_multi(
            self,
            *,
            domain_order: Sequence[Any],
            n_neighbors: int = 20,
            inward_positive: bool = True,
            save_keys=("pseudotime_in", "pseudotime_out", "pseudotime_signed"),
            groupby: Optional[str] = None,
            spatial_smooth: bool = True,  # 新增：是否进行空间平滑
            smooth_radius_factor: float = 1.5,  # 新增：空间平滑半径因子
            min_cluster_size: int = 5,  # 新增：最小聚类大小
            verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
        """
        计算多块区域的双分支拟时序

        参数:
            domain_order: 从内到外的domain顺序（必须包含cambium_label）
            n_neighbors: DPT计算时的近邻数
            inward_positive: True=向内为正值，False=向外为正值
            save_keys: 保存结果的列名
            groupby: 用于分组的列名（None=自动检测连通分量）
            spatial_smooth: 是否进行空间平滑（默认True）
            smooth_radius_factor: 空间平滑半径因子（默认1.5）
            min_cluster_size: 最小聚类大小（小于该值的聚类会被合并）（默认5）
            verbose: 是否打印进度信息
        """
        # 创建副本避免修改原始数据
        adata = self.adata.copy()
        n = adata.n_obs

        # 验证domain_order
        domain_order_str = [str(x) for x in domain_order]
        cambium_label_str = str(self.cambium_label)
        if cambium_label_str not in domain_order_str:
            raise ValueError(f"domain_order必须包含cambium_label={self.cambium_label}")

        # 1. 空间平滑聚类标签（减少孤立点）
        if spatial_smooth:
            self._spatial_smooth_clusters(adata, radius_factor=smooth_radius_factor, verbose=verbose)

        # 2. 移除小聚类（减少噪声）
        if min_cluster_size > 0:
            self._remove_small_clusters(adata, min_size=min_cluster_size, verbose=verbose)

        # 3. 确定空间分块
        comp_labels = self._get_spatial_components(adata, groupby, verbose)

        # 初始化结果数组
        pt_in_all = np.full(n, np.nan, dtype=float)
        pt_out_all = np.full(n, np.nan, dtype=float)
        signed_all = np.full(n, np.nan, dtype=float)
        info_per_comp = {}

        # 4. 对每个空间块分别计算拟时序
        for cid in np.unique(comp_labels):
            idx_comp = np.where(comp_labels == cid)[0]
            sub = adata[idx_comp].copy()

            # 检查是否有形成层
            if not self._has_cambium(sub):
                if verbose:
                    print(f"[Component {cid}] 跳过：无cambium层")
                info_per_comp[cid] = {"skipped": True, "reason": "no cambium"}
                continue

            # 5. 基于domain_order划分内外分支（带空间连续性校正）
            result = self._compute_component_pseudotime(
                sub, idx_comp, domain_order_str, n_neighbors, inward_positive, verbose
            )

            # 6. 空间平滑拟时序结果（进一步提高连续性）
            self._spatial_smooth_pseudotime(sub, result["signed"], radius_factor=1.5)

            # 存储结果
            pt_in_all[idx_comp] = result["pt_in"]
            pt_out_all[idx_comp] = result["pt_out"]
            signed_all[idx_comp] = result["signed"]
            info_per_comp[cid] = result["info"]

        # 7. 保存到adata
        k_in, k_out, k_signed = save_keys
        self.adata.obs[k_in] = pt_in_all
        self.adata.obs[k_out] = pt_out_all
        self.adata.obs[k_signed] = signed_all
        self.adata.obs["pseudotime_component"] = comp_labels

        return comp_labels, info_per_comp

    def _spatial_smooth_clusters(
            self,
            adata: ad.AnnData,
            radius_factor: float = 1.5,
            verbose: bool = True
    ) -> None:
        """对聚类标签进行空间平滑，消除孤立点"""
        X = np.asarray(adata.obsm[self.spatial_key])

        # 计算空间半径
        nn = NearestNeighbors(n_neighbors=min(6, max(2, X.shape[0] - 1))).fit(X)
        dists, _ = nn.kneighbors(X)
        nn1 = np.median(dists[:, 1])
        radius = float(nn1) * radius_factor

        # 构建空间邻域图
        A = NearestNeighbors(radius=radius).fit(X).radius_neighbors_graph(
            X, radius=radius, mode="connectivity"
        )
        A = A.tocsr()

        # 对每个细胞，统计其邻域中的聚类标签
        clusters = adata.obs[self.cluster_key].values
        new_clusters = clusters.copy()
        changed_count = 0

        for i in range(adata.n_obs):
            neighbors = A[i].nonzero()[1]
            if len(neighbors) == 0:
                continue

            # 统计邻域中的聚类标签
            neighbor_clusters = clusters[neighbors]
            unique, counts = np.unique(neighbor_clusters, return_counts=True)
            most_common = unique[np.argmax(counts)]

            # 如果当前标签与最常见标签不同，修正
            if clusters[i] != most_common:
                new_clusters[i] = most_common
                changed_count += 1

        # 更新聚类标签
        adata.obs[self.cluster_key] = new_clusters

        if verbose:
            print(f"空间平滑：{changed_count}个细胞的聚类标签被修正")

    def _remove_small_clusters(
            self,
            adata: ad.AnnData,
            min_size: int = 5,
            verbose: bool = True
    ) -> None:
        """移除太小的聚类簇（合并到最接近的邻域簇）"""
        clusters = adata.obs[self.cluster_key].values
        unique_clusters = np.unique(clusters)
        cluster_sizes = {c: np.sum(clusters == c) for c in unique_clusters}

        # 标记需要移除的小簇
        small_clusters = [c for c in unique_clusters if cluster_sizes[c] < min_size]
        if not small_clusters:
            return

        # 为每个小簇找到最近的大簇
        X = np.asarray(adata.obsm[self.spatial_key])
        cluster_centers = {}
        for c in unique_clusters:
            if c not in small_clusters:
                cluster_centers[c] = np.mean(X[clusters == c], axis=0)

        # 修正小簇
        changed_count = 0
        for c in small_clusters:
            idx = np.where(clusters == c)[0]
            for i in idx:
                # 找到最近的大簇
                dists = [np.linalg.norm(X[i] - center) for center in cluster_centers.values()]
                nearest_cluster = list(cluster_centers.keys())[np.argmin(dists)]
                clusters[i] = nearest_cluster
                changed_count += 1

        # 更新聚类标签
        adata.obs[self.cluster_key] = clusters

        if verbose:
            print(f"移除小簇：{changed_count}个细胞从{len(small_clusters)}个小簇合并到大簇")

    def _get_spatial_components(
            self,
            adata: ad.AnnData,
            groupby: Optional[str],
            verbose: bool
    ) -> np.ndarray:
        """获取空间连通分量"""
        if groupby is not None:
            # 使用指定的分组
            raw = adata.obs[groupby].values
            uniq = {v: i for i, v in enumerate(np.unique(raw))}
            return np.array([uniq[v] for v in raw], dtype=int)
        else:
            # 自动检测连通分量
            X = np.asarray(adata.obsm[self.spatial_key])
            n = adata.n_obs

            # 基于最近邻构建连通图
            k = min(6, max(2, n - 1))
            nn = NearestNeighbors(n_neighbors=k).fit(X)
            dists, _ = nn.kneighbors(X)
            radius = np.median(dists[:, 1]) * 1.5

            A = NearestNeighbors(radius=radius).fit(X).radius_neighbors_graph(
                X, radius=radius, mode="connectivity"
            )
            A = (A + sp.eye(n, format="csr")).tocsr()
            n_comp, comp_labels = sp.csgraph.connected_components(A, directed=False)

            if verbose:
                print(f"检测到 {n_comp} 个空间连通分量")

            return comp_labels

    def _has_cambium(self, adata: ad.AnnData) -> bool:
        """检查是否存在形成层（自动处理类型）"""
        cl = adata.obs[self.cluster_key]
        # 尝试多种类型匹配
        return (
                (cl == self.cambium_label).any() or
                (cl.astype(str) == str(self.cambium_label)).any()
        )

    def _compute_component_pseudotime(
            self, sub: ad.AnnData, idx_comp: np.ndarray,
            domain_order: Sequence[str], n_neighbors: int,
            inward_positive: bool, verbose: bool
    ) -> Dict[str, Any]:
        """计算单个分量的拟时序"""

        # 1. 选择形成层中的平衡根节点
        root_local, idx_cmb_local = self._choose_balanced_root(sub)

        # 2. 基于domain_order划分内外分支
        idx_in_local, idx_out_local = self._assign_branches_by_domain(
            sub, domain_order, idx_cmb_local
        )

        # 3. 空间连续性校正：确保形成层附近区域的连续性
        idx_in_local, idx_out_local = self._correct_branch_continuity(
            sub, idx_in_local, idx_out_local, idx_cmb_local
        )

        # 4. 确保根节点在两个分支中都存在
        if root_local not in idx_in_local:
            idx_in_local = np.union1d(idx_in_local, [root_local])
        if root_local not in idx_out_local:
            idx_out_local = np.union1d(idx_out_local, [root_local])

        # 5. 计算DPT拟时序
        pt_in = self._compute_dpt(sub, idx_in_local, root_local, n_neighbors)
        pt_out = self._compute_dpt(sub, idx_out_local, root_local, n_neighbors)

        # 6. 生成有符号的拟时序
        signed = np.full(sub.n_obs, np.nan, dtype=float)
        signed[idx_in_local] = pt_in[idx_in_local]
        signed[idx_out_local] = -pt_out[idx_out_local]
        signed[idx_cmb_local] = 0.0  # 形成层设为0

        if not inward_positive:
            signed = -signed

        return {
            "pt_in": pt_in,
            "pt_out": pt_out,
            "signed": signed,
            "info": {
                "root_global": int(idx_comp[root_local]),
                "idx_in_global": idx_comp[idx_in_local],
                "idx_out_global": idx_comp[idx_out_local],
                "idx_cambium_global": idx_comp[idx_cmb_local],
            }
        }

    def _correct_branch_continuity(
            self,
            adata: ad.AnnData,
            idx_in: np.ndarray,
            idx_out: np.ndarray,
            idx_cmb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """校正分支连续性，确保形成层附近区域的平滑过渡"""
        X = np.asarray(adata.obsm[self.spatial_key])

        # 计算空间半径（用于确定邻域）
        nn = NearestNeighbors(n_neighbors=min(10, max(2, X.shape[0] - 1))).fit(X)
        dists, _ = nn.kneighbors(X)
        radius = np.median(dists[:, 1]) * 1.5

        # 构建空间邻域图
        A = NearestNeighbors(radius=radius).fit(X).radius_neighbors_graph(
            X, radius=radius, mode="connectivity"
        )
        A = A.tocsr()

        # 1. 检查形成层附近的细胞是否被错误分类
        cambium_neighbors = set()
        for i in idx_cmb:
            neighbors = A[i].nonzero()[1]
            cambium_neighbors.update(neighbors)

        # 2. 检查形成层附近区域的分类一致性
        for i in cambium_neighbors:
            if i in idx_in and i in idx_out:
                # 这个细胞同时属于两个分支，需要修正
                if i in idx_cmb:
                    continue  # 形成层细胞保持不变
                # 检查其邻域中大多数属于哪个分支
                neighbors = A[i].nonzero()[1]
                in_count = sum(1 for n in neighbors if n in idx_in and n not in idx_cmb)
                out_count = sum(1 for n in neighbors if n in idx_out and n not in idx_cmb)

                if in_count > out_count:
                    if i in idx_out:
                        idx_out = np.setdiff1d(idx_out, [i])
                else:
                    if i in idx_in:
                        idx_in = np.setdiff1d(idx_in, [i])

        # 3. 确保形成层附近的区域连续性
        # 对于每个在形成层附近的细胞，检查其邻域中是否有一致的分类
        for i in cambium_neighbors:
            if i in idx_cmb:
                continue

            neighbors = A[i].nonzero()[1]
            in_neighbors = [n for n in neighbors if n in idx_in and n not in idx_cmb]
            out_neighbors = [n for n in neighbors if n in idx_out and n not in idx_cmb]

            if len(in_neighbors) > 0 and len(out_neighbors) > 0:
                # 该细胞的邻域中既有in又有out，需要修正
                if len(in_neighbors) > len(out_neighbors):
                    if i in idx_out:
                        idx_out = np.setdiff1d(idx_out, [i])
                        idx_in = np.union1d(idx_in, [i])
                else:
                    if i in idx_in:
                        idx_in = np.setdiff1d(idx_in, [i])
                        idx_out = np.union1d(idx_out, [i])

        return idx_in, idx_out

    def _spatial_smooth_pseudotime(
            self,
            adata: ad.AnnData,
            pseudotime: np.ndarray,
            radius_factor: float = 1.5,
            verbose: bool = True
    ) -> None:
        """对拟时序结果进行空间平滑，提高连续性"""
        X = np.asarray(adata.obsm[self.spatial_key])

        # 计算空间半径
        nn = NearestNeighbors(n_neighbors=min(10, max(2, X.shape[0] - 1))).fit(X)
        dists, _ = nn.kneighbors(X)
        radius = np.median(dists[:, 1]) * radius_factor

        # 构建空间邻域图
        A = NearestNeighbors(radius=radius).fit(X).radius_neighbors_graph(
            X, radius=radius, mode="connectivity"
        )
        A = A.tocsr()

        # 对每个细胞，用邻域平均值平滑拟时序
        smoothed = pseudotime.copy()
        changed_count = 0

        for i in range(adata.n_obs):
            neighbors = A[i].nonzero()[1]
            if len(neighbors) == 0:
                continue

            # 计算邻域平均值（排除NaN值）
            neighbor_values = pseudotime[neighbors]
            valid_values = neighbor_values[np.isfinite(neighbor_values)]
            if len(valid_values) == 0:
                continue

            avg_value = np.mean(valid_values)

            # 如果当前值与平均值差异较大，进行平滑
            if np.isnan(pseudotime[i]) or abs(pseudotime[i] - avg_value) > 0.5:
                smoothed[i] = avg_value
                changed_count += 1

        # 更新拟时序
        pseudotime[:] = smoothed

        if verbose:
            print(f"拟时序空间平滑：{changed_count}个细胞的拟时序被平滑")

    def _choose_balanced_root(self, adata: ad.AnnData) -> Tuple[int, np.ndarray]:
        """在形成层中选择平衡的根节点"""
        cl = adata.obs[self.cluster_key]
        idx_cmb = np.where(cl == self.cambium_label)[0]

        if idx_cmb.size == 0:
            # 尝试字符串比较
            idx_cmb = np.where(cl.astype(str) == str(self.cambium_label))[0]

        if idx_cmb.size == 0:
            raise ValueError(f"找不到形成层（label={self.cambium_label}）")

        # 基于空间邻居的平衡性选择根节点
        xy = np.asarray(adata.obsm[self.spatial_key])
        center = np.median(xy, axis=0)
        distances = np.linalg.norm(xy - center, axis=1)

        # 选择接近中心的形成层细胞作为根
        cmb_distances = distances[idx_cmb]
        root_local = idx_cmb[np.argmin(cmb_distances)]

        return int(root_local), idx_cmb

    def _assign_branches_by_domain(
            self, adata: ad.AnnData, domain_order: Sequence[str],
            idx_cmb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """基于domain_order划分内外分支"""
        # 创建domain到位置的映射
        order_map = {str(d): i for i, d in enumerate(domain_order)}
        cambium_pos = order_map[str(self.cambium_label)]

        # 获取每个细胞的domain位置
        clusters = adata.obs[self.cluster_key].astype(str).values
        positions = np.array([order_map.get(c, -1) for c in clusters])

        # 划分内外（不包括形成层）
        idx_all = np.arange(adata.n_obs)
        idx_in = idx_all[positions < cambium_pos]  # 内侧：位置小于形成层
        idx_out = idx_all[positions > cambium_pos]  # 外侧：位置大于形成层

        # 将形成层加入两侧（确保连通性）
        idx_in = np.union1d(idx_in, idx_cmb)
        idx_out = np.union1d(idx_out, idx_cmb)

        return idx_in, idx_out

    def _compute_dpt(
            self, adata: ad.AnnData, idx_subset: np.ndarray,
            root_idx: int, n_neighbors: int
    ) -> np.ndarray:
        """计算子集的扩散拟时序（DPT）"""
        # 提取子集的嵌入
        Z_all = np.asarray(adata.obsm[self.embed_key])
        idx_subset = np.asarray(idx_subset, dtype=int)

        # 确认根节点在子集中
        if root_idx not in idx_subset:
            raise ValueError("根节点不在指定子集中")

        # 创建子集AnnData并计算DPT
        root_local = np.where(idx_subset == root_idx)[0][0]
        Z = Z_all[idx_subset]
        ad_sub = ad.AnnData(Z)

        # 计算近邻图
        k = max(2, min(n_neighbors, Z.shape[0] - 1))
        sc.pp.neighbors(ad_sub, n_neighbors=k, use_rep="X")

        # 计算扩散图和DPT
        ad_sub.uns["iroot"] = root_local
        sc.tl.diffmap(ad_sub)
        sc.tl.dpt(ad_sub)

        # 返回完整大小的数组
        pt = np.full(adata.n_obs, np.nan, dtype=float)
        pt[idx_subset] = ad_sub.obs["dpt_pseudotime"].to_numpy()

        return pt

    def plot_pseudotime_map(
            self,
            save_path="./results",
            file_name="pseudotime_map",
            file_format="pdf",
            values: Optional[str] = None,
            colormap="vik",
            vmin=None, vmax=None,
            spot_size=0.45,
            title="Pseudo-Spatiotemporal Map",
            show=True, dpi=300,
    ):
        """绘制空间拟时序图"""
        if values is None:
            raise ValueError("必须指定values参数（如'pseudotime_signed'）")

        # 获取数据
        xy = self.adata.obsm[self.spatial_key].astype(float)
        c = self.adata.obs[values].values.astype(float)

        # 计算点的大小
        radius = self._compute_spot_radius(xy, spot_size)

        # 创建图形
        fig, ax = plt.subplots(figsize=(8, 8))

        # 绘制散点
        patches = [Circle((x, y), radius=radius) for x, y in xy]
        pc = PatchCollection(patches, array=c, cmap=f"cmc.{colormap}",
                             ec="none", linewidth=0)
        if vmin is not None or vmax is not None:
            pc.set_clim(vmin=vmin, vmax=vmax)
        ax.add_collection(pc)

        # 设置坐标轴
        x, y = xy[:, 0], xy[:, 1]
        ax.set_xlim(x.min() - radius, x.max() + radius)
        ax.set_ylim(y.min() - radius, y.max() + radius)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(title, fontsize=14)

        # 添加颜色条
        cbar = fig.colorbar(pc)
        cbar.ax.set_ylabel("Pseudotime", rotation=270, labelpad=15)

        # 保存图像
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, f"{file_name}.{file_format}")
        fig.savefig(full_path, dpi=dpi, bbox_inches="tight")
        print(f"图像已保存: {full_path}")

        if show:
            plt.show()
        plt.close(fig)

        return full_path

    def _compute_spot_radius(self, xy: np.ndarray, spot_size: float) -> float:
        """计算点的半径"""
        if xy.shape[0] <= 3000:
            sample = xy
        else:
            idx = np.random.choice(xy.shape[0], 3000, replace=False)
            sample = xy[idx]

        tree = cKDTree(sample)
        nn_dist = tree.query(sample, k=2)[0][:, 1]
        median_dist = np.median(nn_dist)

        return max(median_dist * spot_size, 1e-6)# class TwoBranchPseudotime:
