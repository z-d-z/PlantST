import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def spatial_pseudotime(adata, start_region):
    # 空间约束降维
    spatial_weight = 0.3  # 空间信息权重
    X_pca = PCA(n_components=20).fit_transform(adata.X)
    X_combined = np.hstack([X_pca, spatial_weight * adata.obsm['spatial']])

    # 构建扩散图
    adata.obsm['X_pca_spatial'] = X_combined
    sc.pp.neighbors(adata, use_rep='X_pca_spatial')
    sc.tl.diffmap(adata)

    # 设置起始区域
    start_cells = np.where(adata.obs['region'] == start_region)[0]
    adata.uns['iroot'] = np.argmax(adata.obsm['X_diffmap'][start_cells, 1])

    # 计算伪时序
    sc.tl.dpt(adata)
    return adata.obs['dpt_pseudotime']

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super(PositionalEncoding, self).__init__()
        self.dim = dim
        self.max_len = None  # 初始时 max_len 为 None，稍后会动态计算
        self.pe = None  # 位置编码矩阵

    def compute_max_len(self, x):
        """
        根据输入坐标数据动态计算 max_len。
        x: (batch_size, 2) -> 每个spot的 (x, y) 绝对坐标
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        max_x = torch.max(x[:, 0]).item()  # 获取 x 轴的最大值
        max_y = torch.max(x[:, 1]).item()  # 获取 y 轴的最大值

        # 选择 x 和 y 的最大值作为 max_len
        self.max_len = int(max(max_x, max_y)) + 1  # 加 1 确保位置编码的索引不会超出范围

        # 创建位置编码矩阵 pe
        pe = torch.zeros(self.max_len, self.dim)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2).float() * (-np.log(self.max_len) / self.dim))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维使用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维使用 cos

        # 只有在 pe 不存在时，才进行注册
        if not hasattr(self, 'pe'):
            self.register_buffer('pe', pe)
        else:
            self.pe = pe  # 如果已存在，则更新 pe

    def forward(self, x):
        """
        x: (batch_size, 2) -> 每个spot的 (x, y) 绝对坐标
        """
        # 如果尚未计算 max_len，则根据输入数据计算 max_len
        if self.max_len is None:
            self.compute_max_len(x)

        # 归一化坐标值，确保不会超出 max_len 范围
        x_normalized = x / torch.tensor(self.max_len, dtype=torch.float32)  # 将坐标归一化到 [0, 1]

        # 归一化后的坐标乘以 max_len，再转为整数索引
        # 在这里我们通过乘以 max_len 来确保坐标的编码位置适配到位置编码矩阵
        x_encoded = self.pe[(x_normalized[:, 0] * (self.max_len - 1)).long()] + self.pe[(x_normalized[:, 1] * (self.max_len - 1)).long()]
        # .detach().cpu().numpy()
        return x_encoded


class CrossModalAttention(nn.Module):
    def __init__(self, input_dim_gene, input_dim_pos, attention_dim, num_heads):
        super(CrossModalAttention, self).__init__()

        # 查询、键、值的线性变换
        self.query = nn.Linear(input_dim_gene, attention_dim)  # 基因表达的查询
        self.key = nn.Linear(input_dim_pos, attention_dim)  # 空间编码的键
        self.value = nn.Linear(input_dim_pos, attention_dim)  # 空间编码的值

        # Multihead Attention 层
        self.attn = nn.MultiheadAttention(attention_dim, num_heads, batch_first=True)

        # 输出的线性变换
        self.fc = nn.Linear(attention_dim, input_dim_gene)

    def forward(self, gene_features, pos_features):
        """
        gene_features: 基因表达特征 (batch_size, num_spots, gene_dim)
        pos_features: 位置编码 (batch_size, num_spots, pos_dim)
        """
        Q = self.query(gene_features)  # (batch_size, num_spots, attention_dim)
        K = self.key(pos_features)  # (batch_size, num_spots, attention_dim)
        V = self.value(pos_features)  # (batch_size, num_spots, attention_dim)

        # 计算注意力
        attn_output, _ = self.attn(Q, K, V)  # attn_output: (batch_size, num_spots, attention_dim)

        # 将注意力输出映射回基因特征空间
        fused_features = self.fc(attn_output)  # (batch_size, num_spots, gene_dim)

        return fused_features

class SpatialGeneModel(nn.Module):
    def __init__(self, input_dim_gene, attention_dim, num_heads, num_layers):
        super(SpatialGeneModel, self).__init__()

        # Transformer Encoder 层
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim_gene, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, fused_features):
        """
        fused_features: (batch_size, num_spots, gene_dim)
        """
        # 使用 Transformer 处理融合后的特征
        output = self.transformer_encoder(fused_features)
        return output

    def train_model(self, fused_features, num_epochs=100):
        """
        训练模型的函数
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()

            # 前向传播
            output = self(fused_features)

            # 假设进行自监督训练，损失函数为重建损失
            loss = criterion(output, fused_features)  # 自监督重建损失
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

        return self


# ring-layer-graph
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
import scanpy as sc
import matplotlib.pyplot as plt

import numpy as np
from sklearn.neighbors import KernelDensity

class SpatialAnalyzer:
    def __init__(self, coord_type, k, angle_weight=0.7, radial_weight=0.3):
        """
        初始化 SpatialAnalyzer 类。

        参数:
        coord_type (str): 坐标类型，'ring' 表示环状，'layer' 表示层状。
        k (int): 每个点寻找的邻居数量。
        angle_weight (float): 角度的权重，默认0.7。
        radial_weight (float): 径向的权重，默认0.3。
        """
        self.coord_type = coord_type
        self.k = k  # 邻居数量
        self.angle_weight = angle_weight  # 角度权重
        self.radial_weight = radial_weight  # 径向权重

    def find_ring_center(self, coords, bandwidth=0.1):
        """
        使用改进的核密度估计（KDE）来寻找空间坐标数据的中心，
        并结合去除离群点来更精确地定位中心点。

        参数:
        coords (ndarray): 空间坐标数据，形状为 (n_samples, 2)。
        bandwidth (float): 核密度估计的带宽参数。

        返回:
        ndarray: 空间坐标数据的中心点。
        """
        # 计算数据的均值作为质心
        centroid = np.mean(coords, axis=0)

        # 去除离群点（根据均值和一定的标准差范围）
        distances = np.linalg.norm(coords - centroid, axis=1)
        threshold = np.mean(distances) + 2 * np.std(distances)  # 去除超出2个标准差的点
        filtered_coords = coords[distances < threshold]

        # 使用核密度估计来更精确地计算中心点
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(filtered_coords)

        # 生成网格
        x_grid = np.linspace(filtered_coords[:, 0].min(), filtered_coords[:, 0].max(), 100)
        y_grid = np.linspace(filtered_coords[:, 1].min(), filtered_coords[:, 1].max(), 100)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

        # 计算密度
        log_dens = kde.score_samples(grid_points)
        dens = np.exp(log_dens)

        # # 返回密度最大的点作为中心点
        # max_idx = np.argmax(dens)
        # return grid_points[max_idx]
        centroid = np.mean(coords, axis=0)
        return centroid

    def plot_center(self, coords, center, output_file='center_plot.png'):
        """
        绘制切片图并标注出中心点。

        参数:
        coords (ndarray): 空间坐标数据，形状为 (n_samples, 2)。
        center (ndarray): 中心点坐标。
        output_file (str): 输出的图片文件名。
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(coords[:, 0], coords[:, 1], c='blue', s=10, label='Spots')  # 绘制所有点
        plt.scatter(center[0], center[1], c='red', s=100, marker='x', label='Center')  # 绘制中心点
        plt.title("Spatial Distribution with Center")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)

        # 保存图像
        plt.savefig(output_file)
        plt.show()

    def hybrid_neighbors(self, coords, center):
        """
        寻找每个点的邻居，自动平衡角度和径向距离，选择最接近的 `k` 个邻居。

        参数:
        coords (ndarray): 空间坐标数据，形状为 (n_samples, 2)。
        center (ndarray): 中心点的坐标。

        返回:
        ndarray: 邻接矩阵，表示每个点与其他点的邻接关系。
        """
        # 计算每个点到中心点的径向距离和角度
        r = np.linalg.norm(coords - center, axis=1)
        theta = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])

        # 邻居矩阵
        neighbors = np.zeros((len(coords), len(coords)))

        # 遍历每个点
        for i in range(len(coords)):
            # 当前点的角度和径向距离
            angle_i = theta[i]
            r_i = r[i]

            # 计算每个点与当前点的角度差和径向差
            angle_diff = np.abs(theta - angle_i)
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)  # 角度差的最小值
            radial_diff = np.abs(r - r_i)  # 径向距离差

            # 根据角度差和径向距离差计算总差距
            total_diff = (self.angle_weight * angle_diff) + (self.radial_weight * radial_diff)

            # 根据总差距排序，选择最接近的k个邻居
            neighbors_idx = np.argsort(total_diff)[1:self.k + 1]  # 排除自己，选择最接近的k个邻居

            # 更新邻接矩阵
            neighbors[i, neighbors_idx] = 1

            # # 打印不为0的邻居
            # if np.any(neighbors[i]):
            #     print(f"点 {i} 的邻居: {neighbors_idx}")

        return neighbors

    def fit(self, adata):
        """
        适配空间数据，寻找邻居关系并将邻接矩阵保存在 adata 中。

        参数:
        adata (AnnData): 包含空间坐标数据的 AnnData 对象。

        返回:
        self: 返回当前对象，便于链式调用。
        """
        # 中心点检测
        if self.coord_type == 'ring':
            self.center = self.find_ring_center(adata.obsm['spatial'])
        elif self.coord_type == 'layer':
            self.center = self.find_layer_center(adata)  # 如果有分层处理，这里是分层中心的计算方法

        # 邻居关系构建
        self.neighbors = self.hybrid_neighbors(
            adata.obsm['spatial'],
            self.center
        )

        # 保存邻接矩阵到 adata
        adata.obsm["physical_distance"] = self.neighbors

        # # 打印中心点
        # print(f"Center found at: {self.center}")
        #
        # # 可视化中心点
        # self.plot_center(adata.obsm['spatial'], self.center, output_file='center_plot.png')

        return self

    def find_layer_center(self, adata):
        """
        示例：可以根据实际需求实现层状数据的中心计算方法。

        参数:
        adata (AnnData): 输入的 AnnData 对象。

        返回:
        ndarray: 层状数据的中心点。
        """
        # 这里可以根据需要实现层状数据的中心点计算
        return np.mean(adata.obsm['spatial'], axis=0)  # 举个例子，返回空间数据的均值




