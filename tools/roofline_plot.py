#!/usr/bin/env python3
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Hardware ceilings physically measured via roofline_bench.cpp
PEAK_AVX2_GFLOPS = 100.0
PEAK_AVX512_GFLOPS = 213.3
PEAK_BW_GBPS = 19.3

# Empirical results for the roofline data points
# Format: (n, compulsory_AI, tiled_gflops, cache_oblivious_gflops)
# Note: AI assumes perfectly cached memory traffic (reads A, reads B, writes C once)
matmul_points = [
    (128, 21.33, 25.73, 23.83),
    (256, 42.67, 23.99, 14.71),
    (512, 85.33, 22.82, 14.14),
    (1024, 170.67, 22.00, 11.58),
]

naive_point = (256, 42.67, 21.74)

fig, ax = plt.subplots(figsize=(9, 6.5))

ai_range = np.logspace(-1, 3, 200)

# The memory bandwidth roof (Performance = AI * Bandwidth)
bw_roof = ai_range * PEAK_BW_GBPS

ax.plot(ai_range, np.minimum(bw_roof, PEAK_AVX512_GFLOPS), color="#888", lw=1.5, ls="--",
        label=f"AVX-512 peak ({PEAK_AVX512_GFLOPS:.0f} GFLOP/s) -- measured, not used by this project's kernels")
ax.plot(ai_range, np.minimum(bw_roof, PEAK_AVX2_GFLOPS), color="#222", lw=2,
        label=f"AVX2 peak ({PEAK_AVX2_GFLOPS:.0f} GFLOP/s) -- the roof this project's kernels target")
ax.axhline(PEAK_AVX2_GFLOPS, color="#222", lw=0.5, ls=":")

xs = [p[1] for p in matmul_points]
tiled_ys = [p[2] for p in matmul_points]
co_ys = [p[3] for p in matmul_points]
ns = [p[0] for p in matmul_points]

ax.plot(xs, tiled_ys, "o-", color="#1f77b4", ms=8, lw=1.5, label="tiled (cache-aware)")
ax.plot(xs, co_ys, "s-", color="#d62728", ms=8, lw=1.5, label="cache_oblivious")
for n, x, y in zip(ns, xs, tiled_ys):
    ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(6, 8), fontsize=8, color="#1f77b4")

ax.plot([naive_point[1]], [naive_point[2]], "^", color="#2ca02c", ms=10,
        label="naive_ikj at n=256 (drawn at compulsory AI -- true AI is lower, see caption)")
ax.annotate("naive_ikj (true position further left)", (naive_point[1], naive_point[2]),
            textcoords="offset points", xytext=(10, 12), fontsize=8, color="#2ca02c")

# Standard roofline charts always use a log-log scale
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Arithmetic intensity (FLOPs / byte)")
ax.set_ylabel("Achieved performance (GFLOP/s)")
ax.set_title("Roofline: matmul kernels vs. this CPU's measured ceilings\n(single-threaded; sandboxed VM, rough not final -- see README)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
ax.set_xlim(1, 300)
ax.set_ylim(5, 300)

fig.tight_layout()
os.makedirs("docs", exist_ok=True)
fig.savefig("docs/roofline.png", dpi=150)
print("wrote docs/roofline.png")