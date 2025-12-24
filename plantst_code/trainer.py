import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from sknetwork.clustering import Louvain
from sklearn.cluster import SpectralClustering, KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
try:
    from .augment import *
except (ImportError, SystemError, ValueError):
    from augment import *
from scipy.spatial.distance import cdist
from torch_sparse import SparseTensor
class train_cluster():
    def __init__(self,
                 adata,
                 processed_data,
                 contrastive_weight,
                 temperature,
                 graph_dict,
                 model,
                 pre_epochs,
                 epochs,
                 k_pos=6,
                 k_neg=6,
                 update_interval=100,
                 gene_sim_threshold_pos=0.7,
                 gene_sim_threshold_neg=0.3,
                 corrupt=0.001,
                 lr=5e-4,
                 weight_decay=1e-4,
                 kl_weight=1,
                 mse_weight=10,
                 bce_kld_weight=0.1,
                 use_gpu=False,
                 ):
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.adata=adata
        # self.processed_data = processed_data
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.data = torch.FloatTensor(processed_data.copy()).to(self.device)
        self.adj = graph_dict['adj_norm'].to(self.device)
        self.adj_label = graph_dict['adj_label'].to(self.device)
        self.norm = graph_dict['norm_value']
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()), lr=lr, weight_decay=weight_decay,amsgrad=True)
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.update_interval = update_interval
        self.gene_sim_threshold_pos = gene_sim_threshold_pos
        self.gene_sim_threshold_neg = gene_sim_threshold_neg
        self.num_spots = self.data.shape[0]
        self.dec_tol = 0
        self.kl_weight = kl_weight
        self.q_stride = 20
        self.mse_weight = mse_weight
        self.bce_kld_weight = bce_kld_weight
        self.corrupt = corrupt

        from sklearn.neighbors import NearestNeighbors
        import numpy as np

        # 1) 空间近邻（用于“近负样本”候选），k_near 可调
        self._coords = np.asarray(self.adata.obsm["spatial"], dtype=np.float32)
        k_near = 16
        nn_spatial = NearestNeighbors(n_neighbors=k_near + 1, metric="euclidean").fit(self._coords)
        # indices[:,0] 是自己，去掉
        self._spatial_knn_idx = nn_spatial.kneighbors(return_distance=False)[:, 1:]  # (N, k_near)

        # 2) 基因表达的“相似 TopM”（用于“远正样本”候选），M 可调，使用 PCA 特征
        gene_X = np.asarray(self.adata.obsm["augment_gene_data_pca"], dtype=np.float32)
        nn_gene = NearestNeighbors(n_neighbors=128 + 1, metric="cosine").fit(gene_X)
        self._gene_top_idx = nn_gene.kneighbors(return_distance=False)[:, 1:]  # (N, 128)

        # 3) 近邻的距离，用于远/近阈值判断（只在需要时再取）
        self._spatial_dists = nn_spatial.kneighbors(return_distance=True)[0][:, 1:]  # (N, k_near)

    # 3) 近邻的距离，用于远/近阈值判断（只在需要时再取）
    # def build_pos_neg_samples_optimized(self, adata, z_current, k_pos=12, k_neg=12, epoch=0, update_interval=50,
    #                                     gene_sim_threshold_pos=0.7, gene_sim_threshold_neg=0.3):
    #     global last_samples_optimized
    #     # 如果不是更新周期，直接返回上一次的采样结果，减少重复计算
    #     if epoch % update_interval != 0 and 'last_samples_optimized' in globals():
    #         return last_samples_optimized
    #
    #     # 预计算：空间距离矩阵（假设空间信息不变）
    #     coords = adata.obsm["spatial"]
    #     spatial_dist = cdist(coords, coords, metric="euclidean")
    #     # 对每个样本计算其邻居的排序索引（不包含自身）
    #     sorted_idx = np.argsort(spatial_dist, axis=1)[:, 1:]
    #
    #     # 预计算：基因表达相似度矩阵（基于 PCA 后的数据，假设不变）
    #     gene_expr = np.array(adata.obsm["augment_gene_data_pca"])
    #     gene_sim_matrix = 1 - pairwise_distances(gene_expr, metric='cosine')
    #     gene_sim_matrix[gene_sim_matrix < 0] = 0
    #     # 对当前 embedding 进行归一化，后续用 np.dot 计算余弦相似度
    #     z_norm = z_current / np.linalg.norm(z_current, axis=1, keepdims=True)
    #
    #     N = adata.n_obs
    #     pos_list = []
    #     neg_list = []
    #
    #     # 根据训练进程动态调整阈值（也可以固定）
    #     adaptive_pos = gene_sim_threshold_pos - 0.01 * (epoch // update_interval)
    #     adaptive_neg = gene_sim_threshold_neg + 0.01 * (epoch // update_interval)
    #
    #     # 遍历每个样本
    #     for i in range(N):
    #         idx_sorted = sorted_idx[i]
    #         # 基于预计算的基因相似度矩阵，取排序后对应的相似度
    #         gene_sims = gene_sim_matrix[i, idx_sorted]
    #
    #         # 正样本：选择相似度高的邻居
    #         pos_candidates = idx_sorted[gene_sims >= adaptive_pos]
    #         if len(pos_candidates) < k_pos:
    #             pos_candidates = np.concatenate([pos_candidates, idx_sorted[:k_pos]])
    #         pos_samples = np.random.choice(pos_candidates, min(k_pos, len(pos_candidates)), replace=False)
    #         pos_list.extend([(i, j) for j in pos_samples])
    #
    #         # 负样本：先选择相似度低的候选，再从中挑选 hardest negatives
    #         neg_candidates = idx_sorted[gene_sims <= adaptive_neg]
    #         if len(neg_candidates) >= k_neg // 2:
    #             # 使用 np.dot 快速计算当前样本和候选负样本间的余弦相似度
    #             emb_sim = np.dot(z_norm[i].reshape(1, -1), z_norm[neg_candidates].T)[0]
    #             # 选择余弦相似度最高的前 k_neg//2 个作为 hard negative
    #             hard_neg = neg_candidates[np.argsort(-emb_sim)[:k_neg // 2]]
    #         else:
    #             hard_neg = neg_candidates
    #
    #         remaining = k_neg - len(hard_neg)
    #         # 如果硬负样本数量不足，从所有邻居中随机补充
    #         if remaining > 0:
    #             random_neg = np.random.choice(idx_sorted, remaining, replace=False)
    #         else:
    #             random_neg = np.array([], dtype=int)
    #         neg_samples = np.concatenate([hard_neg, random_neg])
    #         neg_list.extend([(i, j) for j in neg_samples])
    #
    #     last_samples_optimized = (pos_list, neg_list)
    #     return last_samples_optimized

    def build_pos_neg_samples_optimized(
            self,
            adata,
            z_current,  # torch.Tensor (N, zdim) on device
            k_pos=6, k_neg=6,
            epoch=0, update_interval=100,
            gene_sim_threshold_pos=0.7,
            gene_sim_threshold_neg=0.3,
            far_thresh=60.0, near_thresh=20.0,
            cl_anchor_frac=0.5,  # 只抽取一部分锚点做CL，极大提速
    ):
        # 缓存：只在更新间隔刷新
        if hasattr(self, "_cached_pairs") and (epoch % update_interval != 0):
            return self._cached_pairs

        N = self._coords.shape[0]
        anchors = np.arange(N)
        if cl_anchor_frac < 1.0:
            # 随机采样一部分锚点
            m = max(1, int(N * cl_anchor_frac))
            anchors = np.random.choice(anchors, size=m, replace=False)

        # —— 准备当前表示（用于 hard-negative 打分），全用 GPU，不落到 numpy ——
        with torch.no_grad():
            z_norm = z_current / (z_current.norm(dim=1, keepdim=True) + 1e-8)  # (N, zdim)
            z_norm_cpu = z_norm.detach().cpu()  # 仅小规模索引时拿部分到CPU

        pos_list, neg_list = [], []
        # 近邻候选（近负）：空间 knn
        knn_idx = self._spatial_knn_idx
        # 远正候选：从表达 TopM 里去掉空间近邻，并用阈值筛选
        gene_top = self._gene_top_idx
        coords = self._coords

        for i in anchors:
            # --- 近负候选（空间近邻） ---
            near_cand = knn_idx[i]  # (k_near,)
            # 计算与 near_cand 的“表达相似度”，用 gene_X 的余弦 -> 近似：用 z_norm 来打分（与任务相关）
            sim_near = (z_norm_cpu[i].unsqueeze(0) @ z_norm_cpu[near_cand].T).squeeze(0).numpy()
            # 先按“表达不相似”筛掉：<= neg_thresh
            keep = sim_near <= gene_sim_threshold_neg
            neg_c = near_cand[keep]
            if neg_c.size > 0:
                # 在这些不相似的近邻里，挑“最相似”(embedding上)作为 hardest（越相似越容易混淆）
                hard_order = np.argsort(-sim_near[keep])[:k_neg]
                neg_pick = neg_c[hard_order]
                neg_list.extend([(int(i), int(j)) for j in neg_pick])

            # --- 远正候选（表达 TopM） ---
            far_cand = gene_top[i]  # (M,)
            # 排除太近的点（空间距离 < near_thresh）
            vec = coords[far_cand] - coords[i]
            dist = np.sqrt((vec ** 2).sum(1))
            far_keep = dist > far_thresh
            far_idx = far_cand[far_keep]
            if far_idx.size == 0:
                # 兜底：从非近邻中按 sim 排
                vec2 = coords[far_cand] - coords[i]
                dist2 = np.sqrt((vec2 ** 2).sum(1))
                far_idx = far_cand[dist2 > near_thresh]
                if far_idx.size == 0:
                    continue

            # 用 gene-X 的相似度阈值（近似用 z_norm 或预存的 gene sim 均可；这里直接用 z_norm）
            sim_far = (z_norm_cpu[i].unsqueeze(0) @ z_norm_cpu[far_idx].T).squeeze(0).numpy()
            pos_mask = sim_far >= gene_sim_threshold_pos
            pos_c = far_idx[pos_mask]
            if pos_c.size == 0:
                # 兜底：取分数最高的前 k_pos
                order = np.argsort(-sim_far)[:k_pos]
                pos_c = far_idx[order]
            else:
                if pos_c.size > k_pos:
                    pos_c = np.random.choice(pos_c, size=k_pos, replace=False)

            pos_list.extend([(int(i), int(j)) for j in pos_c])

        self._cached_pairs = (pos_list, neg_list)
        return self._cached_pairs

    def _cl_weight_at(self, epoch):
        warm_epochs = max(1, int(0.3 * self.epochs))
        scale = min(1.0, epoch / warm_epochs)
        return self.contrastive_weight * scale

    def pretrain(
            self,
            grad_down=5,
    ):
        with tqdm(total=int(self.pre_epochs),
                  desc="PlantST trains an initial model 预训练",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for epoch in range(self.pre_epochs):

                self.model.train()
                self.optimizer.zero_grad()
                z, mu, logvar, de_feat, _, feat_x, gnn_z = self.model(self.data, self.adj)
                preds = self.model.dc(z)
                # (pos_spatial, hard_neg_spatial) = self.build_pos_neg_samples_optimized(
                #     adata=self.adata,
                #     z_current=z.detach().cpu().numpy(),
                #     epoch=epoch,
                #     k_pos=6,
                #     k_neg=6,
                #     update_interval=100,
                #     gene_sim_threshold_pos=0.7,
                #     gene_sim_threshold_neg=0.3
                # )
                loss_recon = self.model.PlantST_loss(
                    decoded=de_feat,
                    x=self.data,
                    preds=preds,
                    labels=self.adj_label,
                    mu=mu,
                    logvar=logvar,
                    n_nodes=self.num_spots,
                    norm=self.norm,
                    mask=self.adj_label,
                    mse_weight=self.mse_weight,
                    bce_kld_weight=self.bce_kld_weight,
                )

                # loss_contrast = self.model.multiview_contrastive_loss(z, pos_spatial, hard_neg_spatial, self.temperature)
                # 加入对比学习损失
                # loss = loss_recon + self.contrastive_weight * loss_contrast
                loss = loss_recon
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_down)
                self.optimizer.step()
                pbar.update(1)


    @torch.no_grad()
    #用于生成数据的潜在表示，这些表示可以用于下游分析或聚类。
    def process(
            self,
    ):
        self.model.eval()

        z, _, _, _, q, _, _ = self.model(self.data, self.adj)

        z = z.cpu().detach().numpy()
        q = q.cpu().detach().numpy()

        return z, q

    #包括可选的预训练步骤、聚类初始化（如果使用了如KMeans或Louvain等聚类方法），以及主训练循环。
    #在聚类类型决定的情况下，根据潜在空间的数据点分布更新聚类中心。
    def fit(self,
            cluster_n=20,
            clusterType='Louvain',
            res=1.0,
            pretrain=True,
            ):
        if pretrain:
            self.pretrain()
            pre_z, _ = self.process()

        if clusterType == 'KMeans':
            cluster_method = KMeans(n_clusters=cluster_n, n_init=cluster_n * 2, random_state=88)
            y_pred_last = np.copy(cluster_method.fit_predict(pre_z))
            self.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(self.device)
        elif clusterType == 'Louvain':
            cluster_data = sc.AnnData(pre_z)
            sc.pp.neighbors(cluster_data, n_neighbors=cluster_n)
            sc.tl.louvain(cluster_data, resolution=res)
            y_pred_last = cluster_data.obs['louvain'].astype(int).to_numpy()
            n_clusters = len(np.unique(y_pred_last))
            features = pd.DataFrame(pre_z, index=np.arange(0, pre_z.shape[0]))
            Group = pd.Series(y_pred_last, index=np.arange(0, features.shape[0]), name="Group")
            Mergefeature = pd.concat([features, Group], axis=1)
            cluster_centers_ = np.asarray(Mergefeature.groupby("Group").mean())
            self.model.cluster_layer.data = torch.tensor(cluster_centers_).to(self.device)

        best_loss = float('inf')  # 初始最优损失为无穷大
        # 用于记录训练损失
        train_losses = []
        with tqdm(total=int(self.epochs),
                  desc="PlantST trains a final model",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for epoch in range(self.epochs):
                if epoch % self.q_stride == 0:
                    _, q = self.process()
                    q = self.model.target_distribution(torch.Tensor(q).clone().detach())
                    y_pred = q.cpu().numpy().argmax(1)
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = np.copy(y_pred)
                    self.model.train()
                    if epoch > 0 and delta_label < self.dec_tol:
                        # print('delta_label {:.4}'.format(delta_label), '< tol', self.dec_tol)
                        print('Reached tolerance threshold. Stopping training.')
                        break
                torch.set_grad_enabled(True)
                self.model.train()
                self.optimizer.zero_grad()
                # forward pass
                inputs_coor = self.data.to(self.device)
                z, mu, logvar, de_feat, out_q, feat_x, gnn_z = self.model(Variable(inputs_coor), self.adj)
                preds = self.model.dc(z)
                # loss calculation
                loss_PlantST = self.model.PlantST_loss(
                    decoded=de_feat,
                    x=self.data,
                    preds=preds,
                    labels=self.adj_label,
                    mu=mu,
                    logvar=logvar,
                    n_nodes=self.num_spots,
                    norm=self.norm,
                    mask=self.adj_label,
                    mse_weight=self.mse_weight,
                    bce_kld_weight=self.bce_kld_weight
                )
                pos_spatial, hard_neg_spatial = self.build_pos_neg_samples_optimized(
                    adata=self.adata,
                    z_current=z,  # 直接传 torch.Tensor，省 CPU↔GPU 来回
                    epoch=epoch,
                    k_pos=self.k_pos, k_neg=self.k_neg,
                    update_interval=self.update_interval,
                    gene_sim_threshold_pos=self.gene_sim_threshold_pos,
                    gene_sim_threshold_neg=self.gene_sim_threshold_neg,
                    far_thresh=60.0, near_thresh=20.0,
                    cl_anchor_frac=0.5,  # 比如 50% 锚点参与 CL
                )
                loss_contrast = self.model.multiview_contrastive_loss(z, pos_spatial, hard_neg_spatial, self.temperature)

                loss_kl = F.kl_div(out_q.log(), q.to(self.device))

                # 加入对比学习损失

                # total loss
                loss = loss_PlantST + self.kl_weight * loss_kl + self._cl_weight_at(epoch) * loss_contrast
                # backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
                train_losses.append(loss.item())
                pbar.update(1)
        print("best_loss", best_loss)
        self.model.eval()




from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
from sklearn.metrics import silhouette_score  # 新：嵌入质量监控

class PseudotimeTrainer:
    """
    优化Pseudotime训练器：修复早停，添加scheduler和批处理
    """
    def __init__(self,
                 adata,
                 processed_data,
                 spatial_graph,
                 model,
                 epochs,
                 spatial_regularization_strength=0.2,  # 增加强度改善梯度
                 lr=5e-4,
                 max_patience=100,
                 min_stop=200,
                 batch_size=1024,  # 新：批处理加速
                 use_gpu=True,
                 gpu_id=0,
                 regularization_acceleration=True,
                 edge_subset_sz=1_000_000,
                 embedding_save_path="./embedding.tsv"):
        self.adata = adata
        self.processed_data = processed_data
        self.spatial_graph = spatial_graph
        self.model = model
        self.spatial_reg_strength = spatial_regularization_strength
        self.lr = lr
        self.epochs = epochs
        self.max_patience = max_patience
        self.min_stop = min_stop
        self.reg_acceleration = regularization_acceleration
        self.edge_subset_sz = edge_subset_sz
        self.embedding_save_path = embedding_save_path
        self.batch_size = batch_size

        self.device = f"cuda:{gpu_id}" if use_gpu and torch.cuda.is_available() else "cpu"
        self._prepare_data()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)  # 新：调度器
        self.min_loss = np.inf
        self.patience = 0
        self.training_losses = []

    def _prepare_data(self):
        if hasattr(self.processed_data, "X"):
            X = self.processed_data.X.toarray() if sp.issparse(self.processed_data.X) else np.asarray(self.processed_data.X)
            coords = np.asarray(self.processed_data.obsm.get("spatial", self.adata.obsm["spatial"]))
        else:
            X = np.asarray(self.processed_data, dtype=float)
            coords = np.asarray(self.adata.obsm["spatial"])

        self.n_cells = X.shape[0]
        self.expr = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.coords = torch.tensor(coords, dtype=torch.float32, device=self.device)

        # 处理图：支持多种格式，转换为edge_index
        if isinstance(self.spatial_graph, dict) and "adj_norm" in self.spatial_graph:
            self.edge_index = self.spatial_graph["adj_norm"].to_torch_sparse_coo_tensor().coalesce().indices()
        elif sp.issparse(self.spatial_graph):
            self.edge_index, _ = from_scipy_sparse_matrix(self.spatial_graph)
        else:
            raise ValueError("Unsupported spatial_graph format.")

        self.edge_index = self.edge_index.to(self.device)
        self.model = self.model.to(self.device)

        # 新：批处理DataLoader（加速大数据集）
        self.data = Data(x=self.expr, edge_index=self.edge_index)
        self.loader = DataLoader([self.data], batch_size=self.batch_size, shuffle=True)

    def _compute_spatial_regularization(self, z):
        if self.reg_acceleration:
            m = min(self.edge_subset_sz, self.n_cells)
            idx1 = torch.randint(0, self.n_cells, (m,), device=self.device)
            idx2 = torch.randint(0, self.n_cells, (m,), device=self.device)
            z1, z2 = z[idx1], z[idx2]
            c1, c2 = self.coords[idx1], self.coords[idx2]
            z_dists = torch.norm(z1 - z2, p=2, dim=1) / (torch.max(torch.norm(z1 - z2, p=2, dim=1)) + 1e-8)
            sp_dists = torch.norm(c1 - c2, p=2, dim=1) / (torch.max(torch.norm(c1 - c2, p=2, dim=1)) + 1e-8)
            penalty = torch.mean((1.0 - z_dists) * sp_dists)
        else:
            z_d = torch.cdist(z, z, p=2)
            s_d = torch.cdist(self.coords, self.coords)
            z_dists = z_d / (z_d.max() + 1e-8)
            sp_dists = s_d / (s_d.max() + 1e-8)
            penalty = torch.mean((1.0 - z_dists.flatten()) * sp_dists.flatten())
        return penalty

    def train_step(self):
        self.model.train()
        total_loss = 0
        for batch in self.loader:  # 新：批处理
            self.optimizer.zero_grad()
            z, neg_z, summary = self.model(batch.x, batch.edge_index, self.coords)
            loss = self.model.loss(z, neg_z, summary)
            if self.spatial_reg_strength > 0:
                penalty = self._compute_spatial_regularization(z)
                loss += self.spatial_reg_strength * penalty
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.loader)

    def fit(self):
        with tqdm(total=self.epochs, desc="PlantST trains a final model", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for epoch in range(self.epochs):
                train_loss = self.train_step()
                self.training_losses.append(train_loss)
                self.scheduler.step(train_loss)  # 新：调度lr

                # 修复早停：损失改善时更新
                if train_loss < self.min_loss:
                    self.min_loss = train_loss
                    self.best_params = self.model.state_dict()
                    self.patience = 0
                else:
                    self.patience += 1

                if self.patience > self.max_patience and epoch > self.min_stop:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                pbar.update(1)

        if self.best_params is not None:
            self.model.load_state_dict(self.best_params)

    def process(self):
        print("Processing embeddings...")
        self.model.eval()
        with torch.no_grad():
            z, _, _ = self.model(self.expr, self.edge_index, self.coords)
            self.embedding = z.detach().cpu().numpy()

        # 新：监控嵌入质量
        if "cluster" in self.adata.obs:
            score = silhouette_score(self.embedding, self.adata.obs["cluster"])
            print(f"Embedding silhouette score: {score:.4f}")

        self._save_embedding()
        return self.embedding, self.training_losses

    def _save_embedding(self):
        os.makedirs(os.path.dirname(self.embedding_save_path), exist_ok=True)
        np.savetxt(self.embedding_save_path, self.embedding, delimiter="\t")
        print(f"Embedding saved at {self.embedding_save_path}")
