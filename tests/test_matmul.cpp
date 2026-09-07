#include "qscorer/matmul.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace mm;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "CHECK FAILED: %s at %s:%d\n", #cond,       \
                          __FILE__, __LINE__);                               \
            std::abort();                                                    \
        }                                                                    \
    } while (0)

static bool relative_close(float a, float b, float tol) {
    float diff = std::fabs(a - b);
    float scale = std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
    return diff <= tol * scale;
}

// Correctness across small, odd, and prime sizes -- specifically to stress
// the recursive splitting logic in cache_oblivious, which must handle
// m/2, m-m/2 splits correctly at every level for arbitrary (not just
// power-of-2) sizes.
//
// naive_ikj, tiled, and cache_oblivious are checked for EXACT (bit-
// identical) equality with each other -- not by luck, and not a weaker
// tolerance check. All three share the same "outer k, inner j" structure,
// so a compiler that auto-vectorizes the inner j-loop is vectorizing
// across independent (i,j) output locations, not reassociating any single
// one's k-sum -- confirmed by inspecting the generated assembly rather
// than assumed. cache_oblivious specifically: its K-split "combines" two
// halves by having the second half accumulate directly into the *same*
// running C[i][j] (add_to_c=true) rather than materializing an
// independent value and adding it as a final step, and always processes
// the lower k sub-range before the higher one -- follow that through by
// induction and the whole recursion reduces, for any fixed (i,j), to a
// single accumulator touched in strictly increasing k order, identical to
// naive_ikj's. Checked directly: an isolated version that *does*
// materialize both K-halves independently before adding reassociates on
// ~89% of random trials, confirming this exactness isn't a property of
// float addition in general -- it's specific to this choice.
//
// naive_ijk is compared with a tolerance instead, for a concrete, opposite
// reason: its inner loop is a scalar reduction (many k-terms into one
// C[i][j]), and at -O3 the compiler unrolls it across multiple
// independent accumulator registers to hide FMA latency, then combines
// them at the end -- reassociating that specific sum. Seen directly in
// the generated assembly (repeated vaddss into different registers,
// combined at the end), not inferred from the mismatch alone.
void test_correctness() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);

    const int sizes[] = {1, 2, 3, 5, 7, 13, 17, 31, 33, 63, 65, 100, 127, 200, 500, 777};

    for (int n : sizes) {
        std::vector<float> A(static_cast<size_t>(n) * n), B(static_cast<size_t>(n) * n);
        for (auto& v : A) v = val_dist(rng);
        for (auto& v : B) v = val_dist(rng);

        std::vector<float> C_ijk(static_cast<size_t>(n) * n), C_ikj(static_cast<size_t>(n) * n),
            C_tiled(static_cast<size_t>(n) * n), C_co(static_cast<size_t>(n) * n);

        naive_ijk(A.data(), B.data(), C_ijk.data(), n);
        naive_ikj(A.data(), B.data(), C_ikj.data(), n);
        tiled(A.data(), B.data(), C_tiled.data(), n, /*block_size=*/32);
        cache_oblivious(A.data(), B.data(), C_co.data(), n);

        for (size_t idx = 0; idx < C_ijk.size(); ++idx) {
            CHECK(C_ikj[idx] == C_tiled[idx]);
            CHECK(C_ikj[idx] == C_co[idx]);
            CHECK(relative_close(C_ijk[idx], C_ikj[idx], 1e-3f));
        }
    }
    std::printf("test_correctness: OK across %zu sizes -- ikj/tiled/cache_oblivious bit-exact to "
                "each other; ijk within float tolerance of them (its own reduction reassociates "
                "under -O3's auto-vectorization, see comment above)\n",
                sizeof(sizes) / sizeof(sizes[0]));
}

void bench_block_size_sweep(int n) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    std::vector<float> A(static_cast<size_t>(n) * n), B(static_cast<size_t>(n) * n),
        C(static_cast<size_t>(n) * n);
    for (auto& v : A) v = val_dist(rng);
    for (auto& v : B) v = val_dist(rng);

    const int block_sizes[] = {8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512};
    double flops = 2.0 * n * n * n;

    std::printf("block-size sweep at n=%d (theoretical L1-fit bound for this box's 48KiB L1d, "
                "3 blocks resident: block <= 64):\n",
                n);
    for (int bs : block_sizes) {
        auto t0 = std::chrono::steady_clock::now();
        tiled(A.data(), B.data(), C.data(), n, bs);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        std::printf("  block=%4d: %7.1f ms  (%5.2f GFLOP/s)\n", bs, secs * 1000.0, flops / secs / 1e9);
    }
}

void bench_scaling() {
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    const int BEST_BLOCK = 128; // from the sweep above: peaks/plateaus at 128-256, not the
                                // <=64 the L1-fit arithmetic suggested -- see README

    const int sizes[] = {128, 256, 512, 1024};
    for (int n : sizes) {
        std::vector<float> A(static_cast<size_t>(n) * n), B(static_cast<size_t>(n) * n),
            C(static_cast<size_t>(n) * n);
        for (auto& v : A) v = val_dist(rng);
        for (auto& v : B) v = val_dist(rng);
        double flops = 2.0 * n * n * n;

        std::printf("n=%d:\n", n);
        if (n <= 256) { // naive is slow enough that it's not worth the wait at larger n --
                        // the point it demonstrates is already visible by n=256
            auto t0 = std::chrono::steady_clock::now();
            naive_ijk(A.data(), B.data(), C.data(), n);
            auto t1 = std::chrono::steady_clock::now();
            double s = std::chrono::duration<double>(t1 - t0).count();
            std::printf("  naive_ijk (bad access pattern): %7.1f ms  (%5.2f GFLOP/s)\n", s * 1000,
                        flops / s / 1e9);

            t0 = std::chrono::steady_clock::now();
            naive_ikj(A.data(), B.data(), C.data(), n);
            t1 = std::chrono::steady_clock::now();
            s = std::chrono::duration<double>(t1 - t0).count();
            std::printf("  naive_ikj (loop order only):    %7.1f ms  (%5.2f GFLOP/s)\n", s * 1000,
                        flops / s / 1e9);
        } else { // skip the O(n^3)-but-slow ijk at large n, but ikj is fast enough to
                 // keep -- it's the direct comparison point for whether blocking is
                 // actually buying anything yet at this size
            auto t0 = std::chrono::steady_clock::now();
            naive_ikj(A.data(), B.data(), C.data(), n);
            auto t1 = std::chrono::steady_clock::now();
            double s = std::chrono::duration<double>(t1 - t0).count();
            std::printf("  naive_ikj (loop order only):    %7.1f ms  (%5.2f GFLOP/s)\n", s * 1000,
                        flops / s / 1e9);
        }

        auto t0 = std::chrono::steady_clock::now();
        tiled(A.data(), B.data(), C.data(), n, BEST_BLOCK);
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        std::printf("  tiled (block=%d):               %7.1f ms  (%5.2f GFLOP/s)\n", BEST_BLOCK,
                    s * 1000, flops / s / 1e9);

        t0 = std::chrono::steady_clock::now();
        cache_oblivious(A.data(), B.data(), C.data(), n);
        t1 = std::chrono::steady_clock::now();
        s = std::chrono::duration<double>(t1 - t0).count();
        std::printf("  cache_oblivious:                %7.1f ms  (%5.2f GFLOP/s)\n", s * 1000,
                    flops / s / 1e9);
    }
    std::puts("(sandboxed/VM environment, single shared core, no pinning -- rough, not final)");
}

int main(int argc, char** argv) {
    bool correctness_only = (argc > 1 && std::string(argv[1]) == "--correctness-only");
    test_correctness();
    if (correctness_only) {
        std::puts("all tests passed (correctness-only pass)");
        return 0;
    }
    bench_block_size_sweep(512);
    bench_scaling();
    std::puts("all tests passed");
    return 0;
}
