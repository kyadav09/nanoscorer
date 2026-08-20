# nanoscorer : Zero-Allocation SIMD Micro-Inference Engine

A bare-metal C++ hot path for a market-making/prediction system: feed -> order -> features -> inference -> decision, built for ns-scale latency > throughput. Project aimed to deepen understanding of AI hardware and HFT/quant systems roles independently - the "why" for each design choice below is written for that audience, not just for correctness.

## Philosophy: Memory Wall

This architecture is built on standard processors. Shuttling data between main memory (DRAM) and the CPU is the ultimate latency bottleneck.

While next-generation AI hardware attempts to solve this physically - using memristors and Compute-in-Memory (CIM) crossbars to eliminate the physical distance between storage and ALU - this project sovles it purely in software. By enforcing strict 64-byte struct alignment, using zero-allocation arenas, and designing cache-oblivious matrix multiplication/computation, this engine guarantees that the hot path stays nicely locked inside the CPU's L1 cache. It pushes the standard silicon to its absolute mathematical limits.

## Status

**Component(s) 0/6 done: currently working on the limit order book. **Everything downstream (feature computation, inference kernels, ring buffer connecting them) depends on this: so this component is the starting point. See "Roadmap" below for what's next:

## Design: order-book

Traditional build: sorted structure keyed by price (ie red-black tree, 'std::map', skip list)
This build: a hot path that eliminates unpredictable-latency operations from tree rebalancing and pointer chasing. Instead:

- **Price levels are a flat array**, indexed by 'price - base_price'. Adding, cancelling, or modifying an order's quantity is O(1): direct idx, no search.

- **Best bid / best ask is a hardware bit-scan**, not a tree walk. Each side keeps a bitset (one bit per level, occupied or not); the best price is the highest (bids) or lowest (asks) set bit, found via 'std::countl_zero' / 'std::countr_zero' - one or two once the right 64-bit is loaded.

-