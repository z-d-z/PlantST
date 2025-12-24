import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import Sequential, BatchNorm, InstanceNorm
from torch_geometric.nn import GCNConv, GATConv, Sequential, BatchNorm
from typing import Callable, Iterable, Union, Tuple, Optional
import logging
from torch_geometric.nn import DeepGraphInfomax
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
try:
    from .utils_func import *
except (ImportError, SystemError, ValueError):
    from utils_func import *
class PlantST_model(nn.Module):
    def __init__(self,
                 input_dim,
                 Conv_type,
                 linear_encoder_hidden,
                 linear_decoder_hidden,
                 conv_hidden,
                 p_drop=0.01,
                 dec_cluster_n=15,
                 alpha=0.9,#聚类软分配调整参数，用于计算聚类中心
                 activate="relu",
                 ):
        super(PlantST_model, self).__init__()
        self.input_dim = input_dim
        self.Conv_type = Conv_type
        self.alpha = alpha
        self.conv_hidden = conv_hidden
        self.linear_encoder_hidden = linear_encoder_hidden
        self.linear_decoder_hidden = linear_decoder_hidden
        self.activate = activate
        self.p_drop = p_drop
        self.dec_cluster_n = dec_cluster_n
        # ---- Projection head for contrastive learning (very light) ----
        z_dim = self.linear_encoder_hidden[-1] + self.conv_hidden[-1]
        self.proj = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        current_encoder_dim = self.input_dim
        ## a deep autoencoder network
        self.encoder = nn.Sequential()
        for le in range(len(linear_encoder_hidden)):
            self.encoder.add_module(f'encoder_L{le}',
                                    buildNetwork(current_encoder_dim, self.linear_encoder_hidden[le], self.activate, self.p_drop))
            current_encoder_dim = linear_encoder_hidden[le]
        current_decoder_dim = linear_encoder_hidden[-1] + conv_hidden[-1]

        self.decoder = nn.Sequential()
        for ld in range(len(linear_decoder_hidden)):
            self.decoder.add_module(f'decoder_L{ld}',
                                    buildNetwork(current_decoder_dim, self.linear_decoder_hidden[ld], self.activate, self.p_drop))
            current_decoder_dim = self.linear_decoder_hidden[ld]
        self.decoder.add_module(f'decoder_L{len(self.linear_decoder_hidden)}', nn.Linear(self.linear_decoder_hidden[-1],
                                                                                         self.input_dim))
        # GCN layers
        if self.Conv_type == "GCNConv":
            '''https://arxiv.org/abs/1609.02907'''
            from torch_geometric.nn import GCNConv
            self.conv = Sequential('x, edge_index', [
                (GCNConv(linear_encoder_hidden[-1], conv_hidden[0] * 2), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])
            self.conv_mean = Sequential('x, edge_index', [
                (GCNConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
            self.conv_logvar = Sequential('x, edge_index', [
                (GCNConv(conv_hidden[0] * 2, conv_hidden[-1]), 'x, edge_index -> x1'),
            ])
        if self.Conv_type == "GATConv":
            '''https://arxiv.org/abs/1609.02907'''
            from torch_geometric.nn import GATConv
            self.conv = Sequential('x, edge_index', [
                (GATConv(linear_encoder_hidden[-1], conv_hidden[0] * 2, heads=1, concat=False, dropout=self.p_drop), 'x, edge_index -> x1'),
                BatchNorm(conv_hidden[0] * 2),
                nn.ReLU(inplace=True),
            ])
            self.conv_mean = Sequential('x, edge_index', [
                (GATConv(conv_hidden[0] * 2, conv_hidden[-1], heads=1, concat=False, dropout=self.p_drop), 'x, edge_index -> x1'),
            ])
            self.conv_logvar = Sequential('x, edge_index', [
                (GATConv(conv_hidden[0] * 2, conv_hidden[-1], heads=1, concat=False, dropout=self.p_drop), 'x, edge_index -> x1'),
            ])

        self.dc = InnerProductDecoder(p_drop)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, self.linear_encoder_hidden[-1] + self.conv_hidden[-1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def encode(
            self,
            x,
            adj,
    ):
        feat_x = self.encoder(x)
        conv_x = self.conv(feat_x, adj)
        return self.conv_mean(conv_x, adj), self.conv_logvar(conv_x, adj), feat_x

    def reparameterize(
            self,
            mu,
            logvar,
    ):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def target_distribution(
            self,
            target
    ):
        weight = (target ** 2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def PlantST_loss(
            self,
            decoded,
            x,
            preds,
            labels,
            mu,
            logvar,
            n_nodes,
            norm,
            mask=None,
            mse_weight=10,
            bce_kld_weight=0.1,
    ):
        mse_fun = torch.nn.MSELoss()
        mse_loss = mse_fun(decoded, x)
        if mask is not None:
            preds = preds * mask
            labels = labels * mask

        bce_logits_loss = norm * F.binary_cross_entropy_with_logits(preds, labels)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return mse_weight * mse_loss + bce_kld_weight * (bce_logits_loss + KLD)

    # def multiview_contrastive_loss(self, z, pos_pairs, hard_neg_pairs, temperature=0.1):
    #     z = F.normalize(z, p=2, dim=1)
    #     sim_matrix = torch.mm(z, z.T) / temperature
    #
    #     pos_tensor = torch.tensor(pos_pairs, dtype=torch.long, device=z.device)
    #     pos_sim = sim_matrix[pos_tensor[:, 0], pos_tensor[:, 1]]
    #
    #     hard_neg_tensor = torch.tensor(hard_neg_pairs, dtype=torch.long, device=z.device)
    #     hard_neg_sim = sim_matrix[hard_neg_tensor[:, 0], hard_neg_tensor[:, 1]]
    #
    #     numerator = torch.exp(pos_sim)
    #     denominator = numerator + torch.exp(hard_neg_sim).sum() + 1e-8
    #     loss = torch.mean(-torch.log(numerator / denominator))
    #     return loss
    def multiview_contrastive_loss(self, z, pos_pairs, neg_pairs, temperature=0.5):
        h = F.normalize(self.proj(z), p=2, dim=1)  # (N, 128)
        device = h.device

        # 组织锚点->正/负的索引表
        from collections import defaultdict
        pos_by_i = defaultdict(list)
        neg_by_i = defaultdict(list)
        for i, j in pos_pairs: pos_by_i[i].append(j)
        for i, j in neg_pairs: neg_by_i[i].append(j)

        anchors = list(pos_by_i.keys())
        if len(anchors) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        losses = []
        # 小批处理，避免一次堆太多
        B = 2048  # 每批处理多少个锚点，可按显存改
        for s in range(0, len(anchors), B):
            batch = anchors[s:s + B]
            hi = h[batch]  # (B, 128)

            # 按不等长的正/负集合进行 gather
            # 为了向量化，拼接所有候选索引，记录每个锚点的起止位置
            pos_idx_all, pos_ptr = [], [0]
            neg_idx_all, neg_ptr = [], [0]
            for i in batch:
                pos_idx_all.extend(pos_by_i[i])
                neg_idx_all.extend(neg_by_i[i])
                pos_ptr.append(len(pos_idx_all))
                neg_ptr.append(len(neg_idx_all))

            pos_idx_all = torch.tensor(pos_idx_all, device=device, dtype=torch.long) if len(pos_idx_all) > 0 else None
            neg_idx_all = torch.tensor(neg_idx_all, device=device, dtype=torch.long) if len(neg_idx_all) > 0 else None

            # 计算与所有候选的相似度（仅需 hi 与候选向量的乘法）
            if pos_idx_all is None or neg_idx_all is None:
                continue
            hp = h[pos_idx_all]  # (sumP, 128)
            hn = h[neg_idx_all]  # (sumN, 128)

            # hi 与 hp / hn 的相似度：分块 mm
            # 为了节省显存，这里用逐段切片
            sim_pos_all = torch.sum(hi.repeat_interleave(torch.diff(torch.tensor(pos_ptr)).to(device), dim=0) * hp,
                                    dim=1) / temperature
            sim_neg_all = torch.sum(hi.repeat_interleave(torch.diff(torch.tensor(neg_ptr)).to(device), dim=0) * hn,
                                    dim=1) / temperature

            # 逐锚点做 log-sum-exp
            offset_p, offset_n = 0, 0
            for bi in range(len(batch)):
                P = pos_ptr[bi + 1] - pos_ptr[bi]
                N = neg_ptr[bi + 1] - neg_ptr[bi]
                if P == 0 or N == 0:
                    continue
                pos_logits = sim_pos_all[offset_p:offset_p + P]
                neg_logits = sim_neg_all[offset_n:offset_n + N]
                offset_p += P;
                offset_n += N

                lse_all = torch.logsumexp(torch.cat([pos_logits, neg_logits], dim=0), dim=0)
                lse_pos = torch.logsumexp(pos_logits, dim=0)
                losses.append(-(lse_pos - lse_all))

        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.stack(losses).mean()

    def forward_decoder(self, z):
        for i, layer in enumerate(self.decoder):
            z = layer(z)
            print(f"After decoder layer {i}: z NaN:", torch.isnan(z).any(), "Inf:", torch.isinf(z).any(), "Range:", z.min(), z.max())
            if torch.isnan(z).any():
                raise ValueError(f"NaN detected in decoder layer {i}")
        return z

    def forward(
            self,
            x,
            adj,
    ):
        #自编码器的编码部分 生成一个特征表示 feat_x
        mu, logvar, feat_x = self.encode(x, adj)
        #均值 mu 和对数方差 logvar
        #gnn_z 表示通过图结构信息增强的节点特征
        gnn_z = self.reparameterize(mu, logvar)
        # 拼接矩阵
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z)
        # de_feat = self.forward_decoder(z)
        q = 1.0 / ((1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha) + 1e-8)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, mu, logvar, de_feat, q, feat_x, gnn_z


def buildNetwork(
    in_features,
    out_features,
    activate = "relu",
    p_drop = 0.0
    ):
    net = []
    net.append(nn.Linear(in_features, out_features))
    net.append(BatchNorm(out_features, momentum=0.01, eps=0.001))
    if activate=="relu":
        net.append(nn.ELU())
    elif activate=="sigmoid":
        net.append(nn.Sigmoid())
    if p_drop > 0:
        net.append(nn.Dropout(p_drop))
    return nn.Sequential(*net)

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(
            self,
            dropout,
            act=torch.sigmoid,
    ):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(
            self,
            z,
    ):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj






def corruption(x, edge_index):
    """标准DGI负采样：随机打乱节点特征"""
    return x[torch.randperm(x.size(0))], edge_index

class PseudotimeModel(nn.Module):
    """
    Reverted PseudotimeModel: Removed SpatialAttention and GAT for baseline performance
    """
    def __init__(self, input_dim, hidden_dim=128):  # Keep increased hidden_dim if desired
        super(PseudotimeModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Reverted encoder: Use only GCNConv, no GAT or attention
        self.encoder = GraphEncoder(input_dim, hidden_dim)

        # DGI: Keep explicit corruption
        self.dgi = DeepGraphInfomax(
            hidden_channels=hidden_dim,
            encoder=self.encoder,
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption
        )

    def forward(self, x, edge_index, coords=None):  # Removed coords dependency
        z, neg_z, summary = self.dgi(x, edge_index)
        return z, neg_z, summary  # No spatial attention applied

    def get_embedding(self, x, edge_index, coords=None):
        with torch.no_grad():
            z, _, _ = self.forward(x, edge_index)
        return z

    def loss(self, z, neg_z, summary):
        return self.dgi.loss(z, neg_z, summary)

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False)
        self.prelu1 = nn.PReLU(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=False)  # Reverted to GCNConv
        self.prelu2 = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.prelu1(x)
        x = self.conv2(x, edge_index)
        x = self.prelu2(x)
        return x

