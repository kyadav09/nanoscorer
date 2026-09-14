Markdown
# Nanoscorer: Zero-Allocation SIMD Micro-Inference Engine

A bare-metal C++ hot path for a market-making/prediction system: feed → order book → features → inference → decision. Built for nanosecond-scale latency rather than throughput. 

This portfolio project is aimed at AI hardware and HFT/quant systems engineering roles. Rather than relying on theoretical spec sheets, this engine was built by extracting physical hardware ceilings, managing cache-line footprint, and eliminating unpredictable branches.

## 📖 Architecture & Design Deep-Dive

The exact "why" behind each implementation—including cache profiling, instruction-level constraints, empirical roofline analysis, and data race verifications—is documented extensively.

👉 **[Read the Full Engineering Post-Mortem & Design Spec](docs/design.md)**

## Core Components
* **Flat-Array Order Book:** O(1) allocation/modifications, hardware bit-scans for best bid/ask, and cache-line aligned (`alignas(64)`) intrusive queues.
* **QuickScorer (Branchless GBDT):** Bitmask AND-reductions to eliminate unpredictable data-dependent branches during tree walks.
* **AVX2/SIMD Batch Inference:** Vectorized evaluation of 8 instances simultaneously using native intrinsic compares.
* **Cache-Aware Matmul:** Empirical block-size sweep comparing `ikj` tiled operations vs. recursive cache-oblivious divide-and-conquer.
* **Lock-Free SPSC Ring Buffer:** Wait-free thread handoff built on raw atomics and acquire/release memory barriers.

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
Twelve targets exist: each component has a -O3 -march=native release target, and a sanitized counterpart to catch bugs before trusting the release build (_asan for memory errors, _tsan for thread concurrency verification).

The roofline tools are standalone, not part of this build. They're a one-time-per-machine measurement, not something that needs to rerun on every build:

Bash
g++ -std=c++20 -O3 -march=native -DNDEBUG -Iinclude tools/roofline_bench.cpp -o roofline_bench
./roofline_bench
python3 tools/roofline_plot.py