# PaCMAP Experiments

These experiments on [PaCMAP](https://github.com/YingfanWang/PaCMAP) [1] investigate the role of each pair type and the dynamic weight schedule.

## Structure of the code

```
├── configs/
│   └── config.json           # All 22 experiment configurations
├── scripts/
│   ├── pacmap_source.py      # Original PaCMAP source (from the paper)
│   ├── pacmap_core.py        # Weight schedules, metrics, run function
│   ├── plot_results.py       # Plotting functions
│   └── run_ablations.py      # Main script: runs all experiments + generates figures
├── requirements.txt
└── README.md
```


`pacmap_source.py` - Local copy of the [original PaCMAP library code](https://github.com/YingfanWang/PaCMAP). **This file is not our work**  

`pacmap_core.py` - A script that helps to run PaCMAP with different configurations of weight schedules (since the original API doesn't support this). It includes definitions for weight schedules, evaluation metrics and I/O helpers.  

`plot_results.py` - Plots a bar chart comparing final metrics across configurations as well as grids of scatter plots for the embeddings at each iteration.  

`run_ablations.py` - Main script that loads loads the Mammoth dataset and configurations from `configs/config.json`, runs all experiments, saves results to `results/`, and generates figures (`figures/`). Each experiment is run with both PCA and random initialization.


## Experiments

1. Pair types: disables near, mid-near, or further pairs by zeroing their weight in the loss function.  
2. Weight schedule: replaces the default 3-phase schedule (1000→3→0) with alternatives (constant, no phase 1, no phase 3, reversed).
   
Each configuration is run with both PCA and random initialization on the Mammoth dataset (10K points, 3D) [2].


## Method

We wrap the original PaCMAP library and patch only the `find_weight` function. The optimizer, gradient computation, and all numba-compiled code remain untouched. Pair types are disabled by setting their weight to zero;  pairs are still sampled, but contribute no gradient.


## Metrics

**Random Triplet Accuracy** (global structure): sample $T$ random triplets $(i, j, k)$ and check if the distance ordering is preserved:

$$\text{RTA} = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}\left[\left(d^H_{ij} < d^H_{ik}\right) = \left(d^L_{ij} < d^L_{ik}\right)\right]$$

where $d^H$ and $d^L$ are squared Euclidean distances in high-D and low-D, respectively. A score of 0.5 = random chance.

**Neighbor Preservation** (local structure, $k=10$): fraction of $k$-nearest neighbors preserved in the embedding:

$$\text{NP}_k = \frac{1}{N} \sum_{i=1}^{N} \frac{\left| \mathcal{N}_k^H(i) \cap \mathcal{N}_k^L(i) \right|}{k}$$

where $\mathcal{N}_k^H(i)$ and $\mathcal{N}_k^L(i)$ are the $k$-nearest neighbor sets of point $i$ in high-D and low-D.

## Usage

```
pip install -r requirements.txt
python scripts/run_ablations.py
```

Outputs: `results/` (pickle files) and `figures/` (PNG plots).


## Reference

[1] *Wang, Y., Huang, H., Rudin, C., & Shaposhnik, Y. (2021).* Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMAP, and PaCMAP for Data Visualization. JMLR, 22(201), 1–73.

[2] Smithsonian Institution. Mammuthus primigenius (Blumbach), 2020. https://3d.si.edu/object/3d/mammuthus-primigenius-blumbach:341c96cd-f967-4540-8ed1-d3fc56d31f12.
