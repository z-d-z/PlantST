import os
import psutil
import time
import torch
import math
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import anndata
from pathlib import Path
from sklearn.metrics import pairwise_distances, calinski_harabasz_score
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.spatial import distance
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from typing import Union, Callable
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import h5py
import scipy.sparse as sp
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from torch_sparse import SparseTensor
import os, gzip, pickle
try:
    # 作为包模块导入（from PlantST.PlantST import ...）时走这里
    from .augment import *  # 或显式列出需要的符号
    from .utils_func import *
    from .his_feat import image_feature, image_crop
    from .adj import *
    from .model import *
    from .trainer import *
except (ImportError, SystemError, ValueError):
    # 在 PlantST 目录里“脚本直跑”时退回这里
    from augment import *
    from utils_func import *
    from his_feat import image_feature, image_crop
    from adj import *
    from model import *
    from trainer import *
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
# 配置并执行deep st的主要任务，域标识或数据集成
import anndata as ad
import torch.backends.cudnn as cudnn
import random
import umap


class run():
    def __init__(
            self,
            pre_epochs=None,
            epochs=None,
            save_path="./",
            use_gpu=True,
            seed=42
    ):
        self.save_path = save_path
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.use_gpu = use_gpu
        # 在类初始化时，统一固定随机种子
        self.seed = seed

    # 是用于加载和处理不同平台的空间转录组学数据的函数
    # 该方法根据传入的平台类型读取相应的数据，并且可以选择性地将读取的数据保存到硬盘

    # 指定数据来源的技术平台。目前支持的平台包括 'Visium'、'ST'、'MERFISH'、'slideSeq'、和 'stereoSeq'。
    # 指定包含数据文件的文件夹的路径。
    # 指定具体的数据文件名或文件集。
    # 用于控制是否输出额外的信息，并将原始数据保存到硬盘上。
    def get_adata(
            self,
            platform,
            data_path,
            data_name,
            verbose=True,
    ):
        assert platform in ['Visium', 'ST', 'MERFISH', 'slideSeq', 'stereoSeq']
        if platform in ['Visium', 'ST']:
            if platform == 'Visium':
                adata = read_10X_Visium(os.path.join(data_path, data_name))
            else:
                adata = ReadOldST(os.path.join(data_path, data_name))
        elif platform == 'MERFISH':
            adata = read_merfish(os.path.join(data_path, data_name))
        elif platform == 'slideSeq':
            adata = read_SlideSeq(os.path.join(data_path, data_name))
        # elif platform == 'seqFish':
        # 	adata = read_seqfish(os.path.join(data_path, data_name))
        elif platform == 'stereoSeq':
            adata = read_stereoSeq(os.path.join(data_path, data_name))
        else:
            raise ValueError(f"""\{self.platform!r} does not support.""")
        if verbose:
            save_data_path = Path(os.path.join(self.save_path, "Data", data_name))
            save_data_path.mkdir(parents=True, exist_ok=True)
            adata.write(os.path.join(save_data_path, f'{data_name}_raw.h5ad'), compression="gzip")
        return adata

    # 处理空间转录组数据集中的图像 以提取图像特征
    # 传入的 AnnData 对象，它是空间转录组学数据分析常用的数据结构，可以存储高维度的测量数据以及关于观测和变量的注释。
    # 数据集的名称，用于指定操作和结果文件的保存位置。
    # 指定用于特征提取的卷积神经网络（CNN）类型 ResNet50 是一个广泛使用的深度学习模型，常用于图像分类和特征提取任务。
    # 主成分分析（PCA）用于降维的主成分数量，默认为 50。这是图像特征提取后，为了减少特征维度和突出主要变异源而执行的步骤。
    # 功能执行
    # 构建保存图像裁剪和特征提取结果的目录路径。
    # 确保该路径存在，如果不存在，则创建该目录，包括所有必需的父目录。
    # 对 adata 中的图像进行裁剪，并将结果保存到指定的目录。这个步骤通常涉及根据空间坐标调整图像尺寸或分割图像，以便后续更有效地处理和分析。
    # 这一步骤通过指定的 CNN 模型提取图像特征，然后通过 PCA 进行降维。
    # 提取的特征将存储在 adata 对象中，供后续分析使用。
    def get_image_crop(
            self,
            adata,
            data_name,
            cnnType='ResNet50',
            pca_n_comps=50,
    ):
        # 保存路径设置为 self.save_path/Image_crop/data_name
        save_path_image_crop = Path(os.path.join(self.save_path, 'Image_crop', data_name))
        # 创建目录，确保图像裁剪结果可以保存到指定路径中。
        save_path_image_crop.mkdir(parents=True, exist_ok=True)
        # 对 H&E 切片图像进行裁剪操作，生成每个 spot 对应的小图像。
        adata = image_crop(adata, save_path=save_path_image_crop)
        # 提取裁剪图像的特征:1 加载图像 2 通过卷积神经网络提取特征 3 使用 PCA 降维
        adata = image_feature(adata, pca_components=pca_n_comps, cnnType=cnnType).extract_image_feat()
        return adata

    # 用于增强 AnnData 对象的数据，这是存储空间转录组数据的常用数据结构。这种增强可能包括如归一化基因表达数据、根据空间信息调整数据、以及如果可用的话，整合形态学数据。

    # 将要增强的 AnnData 对象。这个对象通常包含原始基因表达数据，空间坐标，可能还包括形态学数据。
    # 调整数据集中空间相邻点影响力的权重因子。此参数帮助调节邻近点对彼此的影响，对空间归一化至关重要。
    # 在空间邻接矩阵中考虑的最近邻点的数量。这个参数确定数据集中每个点的局部邻域大小。
    # 另一个定义空间考虑范围的参数，可能用于增强过程中的不同上下文或方法，如定义空间聚类或图构建的范围。
    # 在增强过程中用于降维技术的组件数量，如 PCA（主成分分析）。
    # 用于分子距离计算的距离度量类型，如 'cosine'。这影响如何计算分子相似性，进而影响数据归一化和整合。
    # 全局批次距离的距离度量类型，通常为 'correlation'。这与跨不同数据集的批效应校正相关。
    # 是否在增强过程中包括形态学数据。当这类数据可用时，这一点非常重要，因为它可以提供额外的上下文，改善分析。
    # 指定在 AnnData 对象中使用哪个数据子集，通常是 "raw" 数据或可能是转换过或之前归一化过的数据。
    # 指定用于空间数据索引或查询的方法，如 "KDTree"。这一选择影响如何执行空间查询，可以显著影响性能和结果。

    # augment_adata实现了一系列数据变换和增强。该函数应用了基于提供的参数的各种调整，如距离计算、邻居识别、可能的形态学数据整合，以及批效应调整。

    def get_augment(
            self,
            adata,
            spatial_type,
            adjacent_weight=0.4,  # 用于控制直接相邻的 spot 对增强基因表达值的影响程度。
            neighbour_k=8,  # 指定每个 spot 的邻域中最近邻的点数量（基于空间距离）
            spatial_k=20,  # 定义基于空间位置的邻域范围
            n_components=100,  # 增强过程中对基因表达矩阵进行降维处理
            md_dist_type="cosine",  # 用于基因表达的相似性计算，使用余弦距离
            gb_dist_type="correlation",  # 衡量形态学特征之间的相似性，使用相关性距离
            use_morphological=True,  # 将在增强中整合图像特征矩阵
            use_data="raw",  # 使用未经处理的原始基因表达矩阵
    ):
        adata = augment_adata(adata,
                              md_dist_type=md_dist_type,
                              gb_dist_type=gb_dist_type,
                              n_components=n_components,
                              use_morphological=use_morphological,
                              use_data=use_data,
                              neighbour_k=neighbour_k,
                              adjacent_weight=adjacent_weight,
                              spatial_k=spatial_k,
                              spatial_type=spatial_type,
                              )
        print("Step 1: Augment molecule expression is Done!")
        return adata

    # 负责根据给定的空间数据构建图。这个方法通过指定的距离计算类型和参数创建空间邻接矩阵或图，这是空间转录组学分析中理解细胞间相互作用和空间模式的关键步骤。

    # 这个参数接收空间数据，通常包括细胞或测量点的空间坐标。
    # 指定用于图构建的距离计算类型。"BallTree" 是一种常用的高效空间树结构，适用于多维空间的邻近搜索。此参数可以根据具体的空间索引需求选择不同的类型，如 "KDTree" 或其他空间索引结构。
    # 指定在 k-最近邻图中每个点连接的邻居数。此参数控制图的密集程度，k 值较大会使图更加密集，但可能包含更多的噪声或不相关的连接。
    # 用于基于半径的邻居搜索时的截断距离。此参数定义一个半径阈值，只有在此距离内的点才被认为是邻居。这适用于那些距离对连接有实际意义的应用场景。

    # 调用一个假设存在的 graph 函数或类实例，此实例负责接收数据和参数，并执行图的构建。graph(data, distType=distType, k=k, rad_cutoff=rad_cutoff).main() 这行代码执行图构建的主要逻辑，可能包括计算距离、建立索引、连接邻居等。
    def get_graph(
            self,
            adata,
            data,
            distType="BallTree",
            k=6,  #每个节点要连接的邻居数
            rad_cutoff=100,  #用于基于半径的邻居搜索的半径截断值
            n_components=100,
            enh=True,
    ):
        if enh == True:
            gene_correlation = cal_gene_weight(data=adata.obsm["augment_gene_data_pca"],
                                               gene_dist_type='cosine',
                                               n_components=n_components)
            gene_correlation[gene_correlation < 0] = 0  # 更严格的阈值
            graph_dict = graph_enh(data, distType=distType, k=k, rad_cutoff=rad_cutoff,
                                   gene_correlation=gene_correlation).main()
        else:
            graph_dict = graph(data, distType=distType, k=k, rad_cutoff=rad_cutoff).main()
        print("Step 2: Graph computing is Done!")
        return graph_dict

    # 用于寻找空间转录组数据的最优聚类分辨率，它使用了 Leiden 算法，这是一种流行的图数据社区检测方法。该方法通过在不同的分辨率下迭代应用 Leiden 算法，并使用 Calinski-Harabasz 分数来评估聚类质量，从而找到最优的分辨率。
    # 这个方法的目的是通过测试一系列的分辨率来找到最佳的分辨率，这里的“最佳”是基于 Calinski-Harabasz 分数来定义的。Calinski-Harabasz 分数是一种评估聚类质量的方法，它考虑了聚类内的紧密度和聚类间的分离度。该方法试图找到一个分辨率，使得聚类结果在统计意义上最优，即聚类内部元素相似度高，而不同聚类之间差异大。
    # 包含空间转录组数据的 AnnData 对象，应该包含预先计算的邻居图，这是使用 Leiden 算法的前提条件。
    # 分辨率列表，通常通过 np.arange 生成一个分辨率范围，例如从 0.1 到 2.5，步长为 0.01。这个范围内的每个分辨率值都将用于一次 Leiden 算法的聚类尝试。

    # 创建一个空的列表 scores，用于存储每个分辨率下的 Calinski-Harabasz 分数。这个分数是一种评价聚类效果的指标，分数越高，聚类效果通常认为越好。
    # 遍历分辨率列表中的每个值，对 adata 应用 Leiden 聚类算法。sc.tl.leiden(adata, resolution=r) 会根据指定的分辨率对数据进行聚类。
    # 计算每次聚类的 Calinski-Harabasz 分数：calinski_harabasz_score(adata.X, adata.obs["leiden"])，这个分数基于聚类内的聚集程度和聚类间的分离程度计算得出。
    # 将每次迭代的分辨率和对应的分数存储在一个 DataFrame 中
    # 找出具有最高 Calinski-Harabasz 分数的分辨率 best_idx = np.argmax(cl_opt_df["score"])，这表示该分辨率下聚类效果最佳
    # 打印并返回最佳分辨率
    # def _optimize_cluster(
    #         self,
    #         adata,
    #         resolution: list = list(np.arange(0.1, 2.5, 0.01)),
    # ):
    #     scores = []
    #     for r in resolution:
    #         sc.tl.leiden(adata, resolution=r)
    #         s = calinski_harabasz_score(adata.X, adata.obs["leiden"])
    #         scores.append(s)
    #     cl_opt_df = pd.DataFrame({"resolution": resolution, "score": scores})
    #     best_idx = np.argmax(cl_opt_df["score"])
    #     res = cl_opt_df.iloc[best_idx, 0]
    #     print("Best resolution: ", res)
    #     return res
    def _optimize_cluster(
            self,
            adata,
            resolution: list = list(np.arange(0.1, 2.5, 0.01)),
    ):
        scores = []

        # ① 只在函数开始时把 X 变成 dense，一次就够了
        if issparse(adata.X):
            X_dense = adata.X.toarray()
        else:
            X_dense = np.asarray(adata.X)

        # ② 遍历不同分辨率做 Leiden 聚类
        for r in resolution:
            sc.tl.leiden(adata, resolution=r)

            # labels 用当前的 leiden 分群
            labels = adata.obs["leiden"].values

            # ③ 这里用 dense 的 X
            s = calinski_harabasz_score(X_dense, labels)
            scores.append(s)

        cl_opt_df = pd.DataFrame({"resolution": resolution, "score": scores})
        best_idx = np.argmax(cl_opt_df["score"])
        res = cl_opt_df.iloc[best_idx, 0]
        print("Best resolution: ", res)
        return res

    # 为您的空间转录组数据找到一个最适合生成指定数量聚类（或称领域）的 Leiden 算法分辨率。这个方法通过尝试一系列分辨率值，并检查每个分辨率下生成的唯一聚类数目，直到找到能够产生预定数量领域的分辨率为止。以下是该方法的详细解释：
    # _priori_cluster 方法的目标是找到一个分辨率，使得 Leiden 算法产生的聚类数目恰好等于用户预设的领域（或聚类）数目 n_domains。这种方法更加直接，关注于达到一个具体的聚类数量，而不是聚类的质量。这对于那些需要特定数量聚类的分析场景特别有用，例如当聚类数与生物学上已知的类型数相对应时。
    # 旨在为您的空间转录组数据找到一个最适合生成指定数量聚类（或称领域）的 Leiden 算法分辨率。这个方法通过尝试一系列分辨率值，并检查每个分辨率下生成的唯一聚类数目，直到找到能够产生预定数量领域的分辨率为止。以下是该方法的详细解释：

    # 包含空间转录组数据的 AnnData 对象。这个对象应包括必要的图形结构和数据，以便应用 Leiden 聚类算法。
    # 指定期望的领域（或聚类）数量，默认为 7。这是您预期通过聚类得到的独立领域的数量。

    # 方法通过 np.arange(0.1, 2.5, 0.01) 生成一个分辨率的范围，然后对这个列表排序并反转，以从高到低顺序尝试每个分辨率。高分辨率意味着更细致的聚类尝试，通常会生成更多的聚类。
    # 对每个分辨率值应用 Leiden 算法：sc.tl.leiden(adata, random_state=0, resolution=res)。这里包括了一个随机种子（random_state=0）以确保结果的可复现性。
    # 计算每次聚类后的唯一聚类数 这一步统计了在当前分辨率下由 Leiden 算法生成的独立聚类（领域）的数量。
    # 如果某个分辨率下的唯一聚类数等于 n_domains，则停止进一步搜索，因为已找到合适的分辨率。
    # 当找到能产生期望数量领域的分辨率时，打印此分辨率并返回
    def _priori_cluster(
            self,
            adata,
            n_domains,
    ):
        for res in sorted(list(np.arange(0.1, 2.5, 0.01)), reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            if count_unique_leiden == n_domains:
                break
        print("Best resolution: ", res)
        return res

    #为了将多个 AnnData 对象（每个对象可能来自不同的批次或实验）合并成一个单一的 AnnData 对象，并整合相应的图数据结构，以便于后续的统一分析。这个方法特别适用于处理来自不同来源但相关的数据集，使它们可以在统一的分析框架下被比较和分析

    # 包含多个 AnnData 对象的列表，每个对象都代表一个独立的数据集
    # 包含与 adata_list 中每个 AnnData 对象相对应的数据名称的列表，用于标记不同批次或来源的数据
    # 包含与每个 adata 对应的图数据结构的列表，这些图数据结构用于描述数据中的空间或拓扑关系

    # 对于 data_name_list 中的每个数据集名称，方法从 adata_list 和 graph_list 中获取对应的 AnnData 对象和图数据结构。
    # 在 AnnData 对象中添加一个 batch_name 属性，将其设置为相应的数据名称，并将其转换为分类数据类型。
    # 如果是第一个数据集（i == 0），直接将其设置为合并后的数据和图
    # 对于后续的数据集，首先找出变量名（基因名）的交集，以确保合并的数据集在特征上是一致的。
    # 使用 AnnData.concatenate 方法合并当前数据集到之前的合并结果中
    # 调用 combine_graph_dict 函数（这个函数需要自行定义）合并当前图数据结构到之前的合并结果中
    # 在合并后的 AnnData 对象中添加一个 batch 属性，该属性通过转换 batch_name 属性为整数编码得到，以便于在后续分析中使用
    # multiple_adata:合并后包含所有数据集的 AnnData 对象  multiple_graph:合并后的图数据结构，包含了所有数据集的图信息

    def _get_multiple_adata(
            self,
            adata_list,
            data_name_list,
            graph_list,
    ):
        for i in range(len(data_name_list)):
            current_adata = adata_list[i]
            current_adata.obs['batch_name'] = data_name_list[i]
            current_adata.obs['batch_name'] = current_adata.obs['batch_name'].astype('category')
            current_graph = graph_list[i]
            if i == 0:
                multiple_adata = current_adata
                multiple_graph = current_graph
            else:
                var_names = multiple_adata.var_names.intersection(current_adata.var_names)
                multiple_adata = multiple_adata[:, var_names]
                current_adata = current_adata[:, var_names]
                multiple_adata = multiple_adata.concatenate(current_adata)
                multiple_graph = combine_graph_dict(multiple_graph, current_graph)

        multiple_adata.obs["batch"] = np.array(
            pd.Categorical(
                multiple_adata.obs['batch_name'],
                categories=np.unique(multiple_adata.obs['batch_name'])).codes,
            dtype=np.int64,
        )

        return multiple_adata, multiple_graph

    # 用于预处理空间转录组学数据的流程，涉及标准化、对数转换、缩放和主成分分析（PCA）。该方法对输入的 AnnData 对象进行一系列转换，以准备数据进行后续的深入分析

    # 这是一个包含空间转录组学数据的 AnnData 对象。该对象应包含原始基因表达数据以及可能的其他预处理步骤的结果
    # 这是主成分分析（PCA）过程中要保留的主成分数量，默认为200。这个参数控制数据降维的程度，较高的主成分数可以保留更多的原始数据信息，但可能增加后续分析的复杂度和计算需求

    # 将 adata 的当前状态设置为原始数据状态，即 adata.raw = adata。这一步保留了在执行后续操作之前的数据状态，方便恢复或比较
    # 提取并转换 adata.obsm["augment_gene_data"] 中的增强基因数据为 float64 类型，然后更新 adata.X 以用于后续的预处理步骤
    # 使用 sc.pp.normalize_total 对数据进行标准化，使得每个样本的基因表达总和规范到指定的目标总和（target_sum=1）。这一步是为了消除样本间的表达量差异对分析结果的影响
    # 通过 sc.pp.log1p 对标准化后的数据进行对数转换，以稳定表达数据的方差并减少高表达基因的过度影响
    # 使用 sc.pp.scale 对对数转换后的数据进行缩放，即对每个基因进行零均值单位方差的标准化。这一步是为了使得不同基因间的表达量可比，便于进行统计分析
    # 应用 sc.pp.pca 对缩放后的数据进行主成分分析，以降维并提取数据中的主要变异源。指定的 n_comps=pca_n_comps 控制保留的主成分数量

    def data_preprocess_identify(self, adata, pca_n_comps):
        augment_X = adata.obsm["augment_gene_data"].astype(np.float64)
        ad_tmp = sc.AnnData(augment_X)
        sc.pp.normalize_total(ad_tmp, target_sum=1)
        sc.pp.log1p(ad_tmp)
        sc.pp.scale(ad_tmp)
        sc.pp.pca(ad_tmp, n_comps=pca_n_comps)
        adata.obsm["augment_gene_data_pca"] = ad_tmp.obsm["X_pca"].astype(np.float32)
        return adata.obsm["augment_gene_data_pca"]
    def data_preprocess_pseudotime(self, adata, n_top_genes=200):
        means = np.asarray(adata.X.mean(axis=0)).ravel()
        adata = adata[:, means > 0].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='cell_ranger', subset=True)
        sc.pp.pca(adata)
        return adata

    def data_preprocess_ccc(self, adata):
        augment_X = adata.obsm["augment_gene_data"].astype(np.float64)
        ad_tmp = sc.AnnData(augment_X)
        sc.pp.normalize_total(ad_tmp, target_sum=1)
        sc.pp.log1p(ad_tmp)
        enhanced_expr = ad_tmp.X.copy()
        if sparse.issparse(enhanced_expr):
            enhanced_expr = enhanced_expr.toarray()
        return enhanced_expr
    # 负责根据提供的数据和图信息，使用深度学习模型进行训练 此方法支持不同的任务，如领域识别（Identify_Domain）和数据集成（Integration），并根据任务的需求调整模型结构和训练流程

    # 要用于训练的数据，通常是经过预处理的特征矩阵
    # 图信息，用于在模型训练中考虑数据的空间或结构依赖
    # 仅在进行数据集成任务时使用，指定不同数据的域标签
    # 仅在进行数据集成任务时使用，指定域的数量
    # 这些参数用于定义和配置模型结构和训练过程中的权重和层结构，如卷积类型、隐藏层设置、丢失率、各种损失权重等

    # 初始化 PlantST_model，根据提供的参数配置神经网络架构
    # 领域识别 ("Identify_Domain"): 使用基本的 PlantST_model 进行训练，专注于通过图卷积网络识别数据中的空间领域。
    # 数据整合 ("Integration"): 使用 AdversarialNetwork 来扩展 PlantST_model，通过对抗训练整合来自不同领域的数据，这适用于处理多个批次或不同条件下的数据集

    def feature_extration(
            self,
            adata,
            data,
            graph_dict,
            contrastive_weight=None,  # 对比学习权重
            temperature=None,  # 对比学习温度
            task=None,
            k_pos=6,
            k_neg=6,
            update_interval=100,
            gene_sim_threshold_pos=0.7,
            gene_sim_threshold_neg=0.3,
            Conv_type="GCNConv",
            linear_encoder_hidden=[32, 20],
            linear_decoder_hidden=[32],
            conv_hidden=[32, 8],
            p_drop=0.01,
            dec_cluster_n=20,
            kl_weight=1,
            mse_weight=10,
            bce_kld_weight=0.1,
    ):
        global identify_model, PlantST_training
        print("Your task is in full swing, please wait")
        start_time = time.time()
        if task == "identify":
            model = PlantST_model(
                input_dim=data.shape[1],
                Conv_type=Conv_type,
                linear_encoder_hidden=linear_encoder_hidden,
                linear_decoder_hidden=linear_decoder_hidden,
                conv_hidden=conv_hidden,
                p_drop=p_drop,
                dec_cluster_n=dec_cluster_n,
            )
            identify_training = train_cluster(
                adata=adata,
                processed_data=data,
                contrastive_weight=contrastive_weight,
                temperature=temperature,
                graph_dict=graph_dict,
                model=model,
                pre_epochs=self.pre_epochs,
                epochs=self.epochs,
                k_pos=k_pos,
                k_neg=k_neg,
                update_interval=update_interval,
                gene_sim_threshold_pos=gene_sim_threshold_pos,
                gene_sim_threshold_neg=gene_sim_threshold_neg,
                kl_weight=kl_weight,
                mse_weight=mse_weight,
                bce_kld_weight=bce_kld_weight,
                use_gpu=self.use_gpu,
            )
            print("Training new model...")
            # **训练模型**
            identify_training.fit()
            embed, _ = identify_training.process()
            print("Step 3: PlantST training has been Done!")
            print(
                u'Current memory usage：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total time: {total_time / 60 :.2f} minutes")
            print("Your task has been completed, thank you")
            print("Of course, you can also perform downstream analysis on the processed data")
        if task == "pseudotime":
            model = PseudotimeModel(
                input_dim=data.shape[1],
                hidden_dim=25
            )
            trainer = PseudotimeTrainer(
                adata=adata,
                processed_data=data,
                spatial_graph=graph_dict,
                model=model,
                spatial_regularization_strength=0.3,
                lr=5e-4,
                epochs=self.epochs,
                use_gpu=True
            )
            # 训练
            print("Training new model...")
            trainer.fit()
            embed, losses = trainer.process()
            print("Step 3: PlantST training has been Done!")
            print(u'Current memory usage：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total time: {total_time / 60 :.2f} minutes")
            print("Your task has been completed, thank you")
            print("Of course, you can also perform downstream analysis on the processed data")
        return embed


    # 目的是在空间转录组数据上进行聚类分析，以识别不同的生物学领域或区域。这个过程中，根据是否有先验知识（priori 参数）来选择不同的方式确定最佳聚类分辨率，并利用 Leiden 算法进行实际的聚类操作。此外，还包括了一个后处理步骤，以细化聚类结果。
    # 包含空间转录组学数据的 AnnData 对象，其中应包含用于聚类的嵌入表示
    # 期望的领域（聚类）数量
    # 指示是否有关于聚类数量的先验知识 如果为 True，则使用 _priori_cluster 方法寻找能产生指定数量聚类的分辨率；如果为 False，则使用 _optimize_cluster 方法寻找最佳聚类质量的分辨率。
    def get_cluster_data(
            self,
            adata,
            n_domains,
            use_rep,
            priori=True,
    ):
        # 使用 sc.pp.neighbors 计算 adata 的邻居，这是进行 Leiden 聚类前的必要步骤。use_rep='PlantST_embed_final' 表明使用通过深度学习模型得到的嵌入表示作为聚类的基础
        sc.pp.neighbors(adata, use_rep=use_rep)
        if priori:
            # 固定分辨率
            res = self._priori_cluster(adata, n_domains=n_domains)
        else:
            # 寻找最优分辨率
            res = self._optimize_cluster(adata)
        # 使用找到的最佳分辨率 res 在 adata 上执行 Leiden 聚类
        sc.tl.leiden(adata, key_added="domain", resolution=res)
        ######## Strengthen the distribution of points in the model
        # 计算样本点之间的欧氏距离矩阵
        adj_2d = distance.cdist(adata.obsm['spatial'], adata.obsm['spatial'], 'euclidean')
        # 调用 refine 函数，根据样本点的空间分布对初步聚类结果进行细化，以增强模型的空间一致性。这里使用了 "hexagon" 形状假设，这对于某些空间排列数据可能更为合适
        refined_pred = refine(sample_id=adata.obs.index.tolist(),
                              pred=adata.obs["domain"].tolist(), dis=adj_2d, shape="hexagon")

        adata.obs["cluster"] = refined_pred
        # sc.pp.neighbors(adata, use_rep="PlantST_embed", n_neighbors=30)
        # sc.tl.umap(adata, min_dist=0.3, random_state=0)
        # sc.pl.umap(adata, color=["PlantST_refine_domain"])
        return adata

    def evaluate_clustering(self, data, predicted_labels, true_labels=None):
        metrics = {}

        # 基本聚类指标计算
        metrics['silhouette'] = silhouette_score(data, predicted_labels)
        metrics['calinski_harabasz'] = calinski_harabasz_score(data, predicted_labels)
        metrics['davies_bouldin'] = davies_bouldin_score(data, predicted_labels)

        # 如果有真实标签则计算ARI
        if true_labels is not None:
            # 确保标签长度一致
            assert len(predicted_labels) == len(true_labels), "预测标签与真实标签长度不一致"

            # 创建有效样本掩码（过滤NaN）
            valid_mask = ~pd.isnull(predicted_labels) & ~pd.isnull(true_labels)

            # 计算调整兰德指数
            try:
                metrics['ari'] = adjusted_rand_score(true_labels[valid_mask], predicted_labels[valid_mask])
            except Exception as e:
                print(f"计算ARI时发生错误: {str(e)}")
                metrics['ari'] = None

        # 打印结果
        print("=== 聚类评估结果 ===")
        print(f"Silhouette Coefficient: {metrics['silhouette']:.4f}")
        print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz']:.4f}")
        print(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")

        if 'ari' in metrics:
            print(f"Adjusted Rand Index (ARI): {metrics['ari']:.4f}")


