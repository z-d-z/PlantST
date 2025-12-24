# PlantST: Unraveling Plant Tissue Heterogeneity and Developmental Trajectories

**PlantST** is a multimodal graph learning framework tailored for plant spatial transcriptomics. It effectively deciphers spatial domains and reconstructs radial developmental trajectories by leveraging both gene expression and morphological information.

## 🚀 Key Features
- **Topology-Adaptive**: Designed for complex plant geometries (e.g., vascular rings, floral organs).
- **Multimodal**: Integrates histology images (H&E) and gene expression.
- **Spatial Domain Identification**: Accurately partitions distinct tissue regions and delineates sharp boundaries (e.g., cambium vs. xylem) using graph contrastive learning.
- **Trajectory Inference**: Reconstructs continuous radial gradients via spatially constrained pseudotime.

## 🛠 Installation

You can install the required dependencies via pip: 

```bash
pip install -r requirements.txt
