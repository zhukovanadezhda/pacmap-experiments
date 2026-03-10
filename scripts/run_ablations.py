import json
import os
import numpy as np
from pacmap_core import load_mammoth, run_pacmap, save_result
from plot_results import savefig, plot_scatter_grid, plot_bars, plot_lines, plot_snapshots


def main():
    # Load configs and data
    with open("configs/config.json") as f:
        raw = json.load(f)
    configs = [c for group in raw.values() for c in group]

    X, colors = load_mammoth()
    print(f"Mammoth: {X.shape}")

    # Run all experiments
    os.makedirs("results", exist_ok=True)
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {cfg['name']}")
        save_result(run_pacmap(X, cfg))
    np.save("data/mammoth_colors.npy", colors)

    # Generate figures
    print("\n--- Generating figures ---")

    # Ablation A: scatter grid
    savefig(plot_scatter_grid([
        ("PCA", [("A1_full_default","Full"), ("A2_no_midnear","No MN"),
                 ("A3_no_further","No FP"), ("A4_only_near","Only NB"), ("A5_no_near","No NB")]),
        ("Random", [("A1r_full_default_rand","Full"), ("A2r_no_midnear_rand","No MN"),
                    ("A3r_no_further_rand","No FP"), ("A4r_only_near_rand","Only NB"), ("A5r_no_near_rand","No NB")]),
    ], colors, "Which Pair Types Matter?"), "ablation_A_scatter")

    # Ablation A: bar chart
    savefig(plot_bars(
        ["A1_full_default","A2_no_midnear","A3_no_further","A4_only_near","A5_no_near",
         "A1r_full_default_rand","A2r_no_midnear_rand","A3r_no_further_rand","A4r_only_near_rand","A5r_no_near_rand"],
        ["Full\n(PCA)","No MN\n(PCA)","No FP\n(PCA)","Only NB\n(PCA)","No NB\n(PCA)",
         "Full\n(rand)","No MN\n(rand)","No FP\n(rand)","Only NB\n(rand)","No NB\n(rand)"],
        "Final Metrics"), "ablation_A_bars")

    # Ablation A: snapshots
    savefig(plot_snapshots([
        ("A1r_full_default_rand","Full (rand)"), ("A2r_no_midnear_rand","No MN (rand)"),
        ("A4r_only_near_rand","Only NB (rand)"),
    ], colors, "Evolution (Random Init)"), "ablation_A_snapshots")

    # Ablation C: scatter grid
    savefig(plot_scatter_grid([
        ("PCA", [("C1_default_schedule","Default"), ("C2_constant_low","Constant"),
                 ("C3_constant_zero_mn","No MN"), ("C4_no_phase1","No ph.1"),
                 ("C5_no_phase3","No ph.3"), ("C6_reversed","Reversed")]),
        ("Random", [("C1r_default_rand","Default"), ("C2r_constant_low_rand","Constant"),
                    ("C3r_constant_zero_mn_rand","No MN"), ("C4r_no_phase1_rand","No ph.1"),
                    ("C5r_no_phase3_rand","No ph.3"), ("C6r_reversed_rand","Reversed")]),
    ], colors, "Does the Weight Schedule Matter?"), "ablation_C_scatter")

    # Ablation C: line charts
    savefig(plot_lines([
        ("PCA Init", [("C1_default_schedule","Default","#1f77b4"), ("C2_constant_low","Constant","#ff7f0e"),
                      ("C3_constant_zero_mn","No MN","#2ca02c"), ("C4_no_phase1","No ph.1","#d62728"),
                      ("C5_no_phase3","No ph.3","#9467bd"), ("C6_reversed","Reversed","#8c564b")]),
        ("Random Init", [("C1r_default_rand","Default","#1f77b4"), ("C2r_constant_low_rand","Constant","#ff7f0e"),
                         ("C3r_constant_zero_mn_rand","No MN","#2ca02c"), ("C4r_no_phase1_rand","No ph.1","#d62728"),
                         ("C5r_no_phase3_rand","No ph.3","#9467bd"), ("C6r_reversed_rand","Reversed","#8c564b")]),
    ], "Metrics Over Iterations"), "ablation_C_lines")

    # Ablation C: snapshots PCA
    savefig(plot_snapshots([
        ("C1_default_schedule","Default (PCA)"), ("C4_no_phase1","No ph.1 (PCA)"),
        ("C6_reversed","Reversed (PCA)"),
    ], colors, "Evolution (PCA Init)"), "ablation_C_snapshots_pca")

    # Ablation C: snapshots random
    savefig(plot_snapshots([
        ("C1r_default_rand","Default (rand)"), ("C2r_constant_low_rand","Constant (rand)"),
        ("C6r_reversed_rand","Reversed (rand)"),
    ], colors, "Evolution (Random Init)"), "ablation_C_snapshots_rand")

    print("\nAll done!")

if __name__ == "__main__":
    main()