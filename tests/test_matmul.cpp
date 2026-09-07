#include "qscorer/matmul.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace mm;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "CHECK FAILED: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
            std::abort(); \
        } \
    } while(0) \

static bool relative_close(float a, float b, float tol) {
    float diff = std::fabs(a - b);
    float scale = std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
    return diff <= tol * scale;
}

// validate correctness across varied, odd, and prime matrix sizes.
// ikj, tiled, and cache_oblivious match bit-exact (shared k-sum reduction order).
// ijk uses tolerance check due to -O3 auto-vectorization reassociating reduction sums.
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
    std::printf("test_correctness: OK across %zu sizes -- ikj/tiled/cache_oblivious bit-exact; ijk within float tolerance.\n", sizeof(sizes) / sizeof(sizes[0]));
}

void bench_block_size_sweep(int n) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    std::vector<float> A(static_cast<size_t>(n) * n), B(static_cast<size_t>(n) * n), C(static_cast<size_t>(n) * n);
    for (auto& v : A) v = val_dist(rng);
    for (auto& v : B) v = val_dist(rng);

    const int block_sizes[] = {8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512};
    double flops = 2.0 * n * n * n;

    std::printf("block-size sweep at n=%d:\n", n);
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
    const int BEST_BLOCK = 128; // empirically optimal block size from sweep

    const int sizes[] = {128, 256, 512, 1024};
    for (int n : sizes) {
        std::vector<float> A(static_cast<size_t>(n) * n), B(static_cast<size_t>(n) * n), C(static_cast<size_t>(n) * n);
        for (auto& v : A) v = val_dist(rng);
        for (auto& v : B) v = val_dist(rng);
        double flops = 2.0 * n * n * n;

        std::printf("n=%d:\n", n);
        if (n <= 256) {
            auto t0 = std::chrono::steady_clock::now();
            naive_ijk(A.data(), B.data(), C.data(), n);
            auto t1 = std::chrono::steady_clock::now();
            double s = std::chrono::duration<double>(t1 - t0).count();
            std::printf("  naive_ijk (bad access): %7.1f ms  (%5.2f GFLOP/s)\n", s * 1000, flops / s / 1e9);

            t0 = std::chrono::steady_clock::now();
            naive_ikj(A.data(), B.data(), C.data(), n);
            t1 = std::chrono::steady_clock::now();
            s = std::chrono::duration<double>(t1 - t0).count();
            std::printf("  naive_ikj (loop order): %7.1f ms  (%5.2f GFLOP/s)\n", s * 1000, flops / s / 1e9);
        } else {
            // Skip slow O(n^3) naive_ijk at larger scales
            auto t0 = std::chrono::steady_clock::now();
            naive_ikj(A.data(), B.data(), C.data(), n);
            auto t1 = std::chrono::steady_clock::now();
            double s = std::chrono::duration<double>(t1 - t0).count();
            std::printf("  naive_ikj (loop order): %7.1f ms  (%5.2f GFLOP/s)\n", s * 1000, flops / s / 1e9);
        }

        auto t0 = std::chrono::steady_clock::now();
        tiled(A.data(), B.data(), C.data(), n, BEST_BLOCK);
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        std::printf("  tiled (block=%d):       %7.1f ms  (%5.2f GFLOP/s)\n", BEST_BLOCK, s * 1000, flops / s / 1e9);

        t0 = std::chrono::steady_clock::now();
        cache_oblivious(A.data(), B.data(), C.data(), n);
        t1 = std::chrono::steady_clock::now();
        s = std::chrono::duration<double>(t1 - t0).count();
        std::printf("  cache_oblivious:        %7.1f ms  (%5.2f GFLOP/s)\n", s * 1000, flops / s / 1e9);
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