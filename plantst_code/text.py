import os
from PlantST import run
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc

data_path = "../data/DLPFC" #### to your path
data_name = '151673' #### project name
save_path = "../Results" #### save path
n_domains = 7 ###### the number of spatial domains.

deepen = run(save_path = save_path,
	task = "Identify_Domain", #### PlantST includes two tasks, one is "Identify_Domain" and the other is "Integration"
	pre_epochs = 800, ####  choose the number of training
	epochs = 1000, #### choose the number of training
	use_gpu = True)
###### Read in 10x Visium data, or user can read in themselves.
adata = deepen.get_adata(platform="Visium", data_path=data_path, data_name=data_name)
###### Segment the Morphological Image
adata = deepen.get_image_crop(adata, data_name=data_name)

###### Data augmentation. spatial_type includes three kinds of "KDTree", "BallTree" and "LinearRegress", among which "LinearRegress"
###### is only applicable to 10x visium and the remaining omics selects the other two.
###### "use_morphological" defines whether to use morphological images.
adata = deepen.get_augment(adata, spatial_type="LinearRegress", use_morphological=True)

###### Build graphs. "distType" includes "KDTree", "BallTree", "kneighbors_graph", "Radius", etc., see adj.py
graph_dict = deepen.get_graph(adata.obsm["spatial"], distType ="BallTree")

###### Enhanced data preprocessing
data = deepen.data_preprocess_identify(adata, pca_n_comps=200)

###### Training models
PlantST_embed = deepen._fit(
		data = data,
		graph_dict = graph_dict,)
###### PlantST outputs
adata.obsm["PlantST_embed"] = PlantST_embed

###### Define the number of space domains, and the model can also be customized. If it is a model custom priori = False.
adata = deepen.get_cluster_data(adata, n_domains=n_domains, priori = True)

###### Spatial localization map of the spatial domain
sc.pl.spatial(adata, color='PlantST_refine_domain', frameon = False, spot_size=150)
plt.savefig(os.path.join(save_path, f'{data_name}_domains.pdf'), bbox_inches='tight', dpi=300)