#!/usr/bin/env python3

# @Author: ChangXu
# @E-mail: xuchang0214@163.com
# @Last Modified by:   ChangXu
# @Last Modified time: 2021-04-22 08:42:54 23:22:34
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors, KDTree, BallTree
import h5py
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import os, gzip, pickle

class MultiModalAttentionFusion(nn.Module):
	def __init__(self,
				 embed_dim=64,
				 num_heads=4,
				 dropout=0.1):
		super().__init__()
		self.embed_dim = embed_dim
		self.num_heads = num_heads

		# 设备自动检测（CPU/GPU）
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# 双模态特征编码层
		self.sim_encoder = nn.Sequential(
			nn.Linear(2, embed_dim),
			nn.ReLU(),
			nn.Dropout(dropout)
		).to(self.device)

		# 多头注意力机制
		self.attention = nn.MultiheadAttention(
			embed_dim=embed_dim,
			num_heads=num_heads,
			dropout=dropout,
			batch_first=True
		).to(self.device)

		# 权重生成器
		self.weight_generator = nn.Sequential(
			nn.Linear(embed_dim, 32),
			nn.ReLU(),
			nn.Linear(32, 1),
			nn.Sigmoid()
		).to(self.device)

	def forward(self, gene_sim, morph_sim):
		"""
        输入:
            gene_sim: 基因相似性矩阵 [N, N] (numpy.ndarray)
            morph_sim: 形态学相似性矩阵 [N, N] (numpy.ndarray)
        输出:
            weight_matrix: 融合权重矩阵 [N, N] (torch.Tensor)
        """
		# 转换numpy输入为PyTorch张量，并确保在正确设备上
		if isinstance(gene_sim, np.ndarray):
			gene_sim = torch.from_numpy(gene_sim).float().to(self.device)
		if isinstance(morph_sim, np.ndarray):
			morph_sim = torch.from_numpy(morph_sim).float().to(self.device)

		# 输入验证
		assert gene_sim.shape == morph_sim.shape, "输入矩阵维度必须相同"
		assert len(gene_sim.shape) == 2, "输入必须是二维矩阵"

		N = gene_sim.size(0)  # 或 gene_sim.shape[0]

		# 拼接双模态相似性矩阵 [N, N, 2]
		combined = torch.stack([gene_sim, morph_sim], dim=-1)

		# 特征编码 [N, N, 2] → [N, N, embed_dim]
		encoded = self.sim_encoder(combined)

		# 调整注意力输入形状 [N*N, 1, embed_dim]
		attn_input = encoded.view(-1, 1, self.embed_dim)

		# 自注意力计算
		attn_output, _ = self.attention(
			query=attn_input,
			key=attn_input,
			value=attn_input
		)

		# 恢复形状并生成权重
		attn_output = attn_output.view(N, N, self.embed_dim)
		weights = self.weight_generator(attn_output).squeeze(-1)
		sym_weights = (weights + weights.T) / 2  # 对称化

		return sym_weights.detach().cpu().numpy()

def cal_spatial_weight(
	data,
	spatial_type,
	spatial_k,
	):
	if spatial_type == "NearestNeighbors":
		nbrs = NearestNeighbors(n_neighbors = spatial_k+1, algorithm ='ball_tree').fit(data)
		_, indices = nbrs.kneighbors(data)
	elif spatial_type == "KDTree":
		tree = KDTree(data, leaf_size=2) 
		_, indices = tree.query(data, k = spatial_k+1)
	elif spatial_type == "BallTree":
		tree = BallTree(data, leaf_size=2)
		_, indices = tree.query(data, k = spatial_k+1)
	# 去除每个点自身的索引，只保留其他的邻居
	indices = indices[:, 1:]
	# 初始化一个全零矩阵 2687*2687
	spatial_weight = np.zeros((data.shape[0], data.shape[0]))
	print("Starting to fill spatial_weight matrix.")
	for i in range(indices.shape[0]):
		ind = indices[i]
		for j in ind:
			spatial_weight[i][j] = 1
	return spatial_weight

def cal_gene_weight(data, n_components, gene_dist_type):
	if isinstance(data, csr_matrix):
		data = data.toarray()
	if data.shape[1] > 500:
		pca = PCA(n_components = n_components)
		data = pca.fit_transform(data)
		gene_correlation = 1 - pairwise_distances(data, metric = gene_dist_type)
	else:
		gene_correlation = 1 - pairwise_distances(data, metric = gene_dist_type)
	return gene_correlation


def cal_weight_matrix(
		adata,
		spatial_type,# 用于计算空间距离的算法
		spatial_k,# 计算物理距离时的最近邻点数
		md_dist_type="cosine",# 形态学特征相似性计算的距离类型
		gb_dist_type="correlation",# 基因表达相关性计算的距离类型
		n_components = 100,# 主成分分析的成分数量
		use_morphological = True,# 是否使用形态学特征
		):
	if use_morphological:
		if spatial_type == "LinearRegress":
			img_row = adata.obs["imagerow"]
			img_col = adata.obs["imagecol"]
			array_row = adata.obs["array_row"]
			array_col = adata.obs["array_col"]
			rate = 3
			reg_row = LinearRegression().fit(array_row.values.reshape(-1, 1), img_row)
			reg_col = LinearRegression().fit(array_col.values.reshape(-1, 1), img_col)
			physical_distance = pairwise_distances(
									adata.obs[["imagecol", "imagerow"]], 
								  	metric="euclidean")
			unit = math.sqrt(reg_row.coef_ ** 2 + reg_col.coef_ ** 2)
			physical_distance = np.where(physical_distance >= rate * unit, 0, 1)
		else:
			physical_distance = cal_spatial_weight(adata.obsm['spatial'], spatial_k = spatial_k, spatial_type = spatial_type)

	else:
		physical_distance = cal_spatial_weight(adata.obsm['spatial'], spatial_k = spatial_k, spatial_type = spatial_type)
	print("Physical distance calculting Done!")
	print("The number of nearest tie neighbors in physical distance is: {}".format(physical_distance.sum()/adata.shape[0]))
	
	# 基因表达相关性计算
	gene_correlation = cal_gene_weight(data = adata.X.copy(), 
											gene_dist_type = gb_dist_type, 
											n_components = n_components)
	gene_correlation[gene_correlation < 0 ] = 0
	print("Gene correlation calculting Done!")

	# 形态学相似性计算
	if use_morphological: 
		morphological_similarity = 1 - pairwise_distances(np.array(adata.obsm["image_feat_pca"]), metric = md_dist_type)
		morphological_similarity[morphological_similarity < 0] = 0

		print("Morphological similarity calculting Done!")

		# 计算形态学相似性和基因相似性的方差
		var_morph = np.var(morphological_similarity)
		var_gene = np.var(gene_correlation)

		# 计算动态权重参数 a
		a = var_morph / (var_morph + var_gene + 1e-8)  # 避免除 0

		# 计算融合矩阵
		adata.obsm["weights_matrix_all"] =  ((morphological_similarity * a + gene_correlation * (1 - a))
											 # * physical_distance
											 )
		# adata.obsm["weights_matrix_all"] =   physical_distance * gene_correlation * morphological_similarity
		adata.obsm["morphological_similarity"] = morphological_similarity
		adata.obsm["gene_correlation"] = gene_correlation
		print("The weight result of image feature is added to adata.obsm['weights_matrix_all'] !")
	else:
		adata.obsm["weights_matrix_all"] = (gene_correlation * physical_distance)
		adata.obsm["gene_correlation"] = gene_correlation
		print("The weight result of image feature is added to adata.obsm['weights_matrix_all'] !")
	return adata

def find_adjacent_spot(
	adata,
	use_data ,
	neighbour_k ,
	verbose = False,
	):
	# 判断矩阵结构
	if use_data == "raw":
		if isinstance(adata.X, csr_matrix):
			gene_matrix = adata.X.toarray()
		elif isinstance(adata.X, np.ndarray):
			gene_matrix = adata.X
		elif isinstance(adata.X, pd.Dataframe):
			gene_matrix = adata.X.values
		else:
			raise ValueError(f"""{type(adata.X)} is not a valid type.""")
	else:
		gene_matrix = adata.obsm[use_data]
	weights_list = [] # 存储每个spot的邻近权重
	final_coordinates = [] # 存储加权后的基因表达数据
	# 打印进度条
	with tqdm(total=len(adata), desc="Find adjacent spots of each spot",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]",) as pbar:

		# 用于遍历 adata 的所有spots,adata.shape[0] 返回 adata 的行数，即spots的数量
		for i in range(adata.shape[0]):
			# 找到临近的spot
			# argsort() 对 weights_matrix_all 中第 i 行进行排序从小到大，并返回排序后的索引数组
			# [-neighbour_k:] 选择排序后最后 neighbour_k 个索引，即权重最大的邻近spots的索引。
			# [:neighbour_k-1] 从这些索引中再取前 neighbour_k-1 个，通常用于排除自身（如果自身是最高权重之一的情况）。
			current_spot = adata.obsm['weights_matrix_all'][i].argsort()[-neighbour_k:][:neighbour_k-1]
			# 获取选定邻近spots的权重
			spot_weight = adata.obsm['weights_matrix_all'][i][current_spot]
			# 从基因表达矩阵（gene_matrix）中获取这些邻近spots的基因表达数据。
			spot_matrix = gene_matrix[current_spot]
			# 权重加权和计算
			if spot_weight.sum() > 0:
				# 权重归一化：如果权重总和大于0，先对权重进行归一化处理
				spot_weight_scaled = (spot_weight / spot_weight.sum())
				# 将归一化后的权重添加到 weights_list 列表
				weights_list.append(spot_weight_scaled)
				# 使用 np.multiply 对归一化权重和相应的基因表达数据进行元素乘法，然后通过 np.sum 沿着行（axis=0）求和，得到加权平均的基因表达数据。
				spot_matrix_scaled = np.multiply(spot_weight_scaled.reshape(-1,1), spot_matrix)
				#
				spot_matrix_final = np.sum(spot_matrix_scaled, axis=0)
			else:
				spot_matrix_final = np.zeros(gene_matrix.shape[1])
				weights_list.append(np.zeros(len(current_spot)))
			final_coordinates.append(spot_matrix_final)
			pbar.update(1)
		adata.obsm['adjacent_data'] = np.array(final_coordinates)
		if verbose:
			adata.obsm['adjacent_weight'] = np.array(weights_list)
		return adata


def augment_gene_data(
	adata,
	adjacent_weight,
	):
	if isinstance(adata.X, np.ndarray):
		adata.obsm["augment_gene_data"] =  adata.X + adjacent_weight * adata.obsm["adjacent_data"].astype(float)
	elif isinstance(adata.X, csr_matrix):
		adata.obsm["augment_gene_data"] = adata.X.toarray() + adjacent_weight * adata.obsm["adjacent_data"].astype(float)
	return adata

def augment_adata(
	adata,
	spatial_type ,
	spatial_k ,# 空间邻域大小 邻居距离小于30
	adjacent_weight ,# 空间位置矩阵的在增强过程中的占比
	md_dist_type ,# 基因表达距离的度量方法 适合数据向量方向更重要的情况
	gb_dist_type ,# 图像形态学特征距离的度量方法 评估形态特征之间的统计相关性
	n_components ,# 主成分分析（PCA）降维后的特征维度
	use_morphological ,# 是否使用形态学特征参与基因表达增强计算
	use_data ,# 指定增强基因表达时使用的基因数据 raw：原始矩阵  log：对数变换后的矩阵
	neighbour_k ,# 每个 spot 的邻域大小 可以有四个点
	):
	# 结合基因表达相似性和形态学特征相似性还有距离矩阵，生成最终的权重矩阵
	adata = cal_weight_matrix(
				adata,
				md_dist_type = md_dist_type,
				gb_dist_type = gb_dist_type,
				n_components = n_components,
				use_morphological = use_morphological,
				spatial_k = spatial_k,
				spatial_type = spatial_type,
				)
	# 加权基因表达矩阵 可以结合上下文 结合多少个点看neighbour_k数量
	adata = find_adjacent_spot(adata,
				use_data = use_data,
				neighbour_k = neighbour_k)
	# 根据邻近点的信息增强基因表达数据 将增强后的基因表达矩阵存储到 adata 中，用于后续分析
	adata = augment_gene_data(adata,
				adjacent_weight = adjacent_weight)
	return adata




