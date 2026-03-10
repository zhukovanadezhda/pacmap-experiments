# PaCMAP Experiments

These experiments on PaCMAP [1] investigate the role of each pair type and the dynamic weight schedule.

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

## Experiments

1. Pair types: disables near, mid-near, or further pairs by zeroing their weight in the loss function.  
2. Weight schedule: replaces the default 3-phase schedule (1000→3→0) with alternatives (constant, no phase 1, no phase 3, reversed).
   
Each configuration is run with both PCA and random initialization on the Mammoth dataset (10K points, 3D) [2].


## Method

We wrap the original PaCMAP library and patch only the find_weight function. The optimizer, gradient computation, and all numba-compiled code remain untouched. Pair types are disabled by setting their weight to zero;  pairs are still sampled, but contribute no gradient.


## Metrics

Random Triplet Accuracy — global structure: fraction of random triplets whose distance ordering is preserved  
Neighbor Preservation (k=10) — local structure: fraction of k-NN overlap between high-D and 2D


## Usage

```
pip install -r requirements.txt
python scripts/run_ablations.py
```

Outputs: `results/` (pickle files) and `figures/` (PNG plots).


## Reference

[1] *Wang, Y., Huang, H., Rudin, C., & Shaposhnik, Y. (2021).* Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMAP, and PaCMAP for Data Visualization. JMLR, 22(201), 1–73.

[2] Smithsonian Institution. Mammuthus primigenius (Blumbach), 2020. https://3d.si.edu/object/3d/mammuthus-primigenius-blumbach:341c96cd-f967-4540-8ed1-d3fc56d31f12.
