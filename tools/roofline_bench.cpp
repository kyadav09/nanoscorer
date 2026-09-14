#include "matmul/matmul.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <immintrin.h>
#include <random>
#include <vector>

using namespace mm;

template <int NUM_ACC>
double measure_avx2_peak_gflops(double seconds) {
    __m256 acc[NUM_ACC];
    __m256 mul = _mm256_set1_ps(1.0000001f);
    __m256 add = _mm256_set1_ps(1e-6f);
    for (int i = 0; i < NUM_ACC; ++i) 
        acc[i] = _mm256_set1_ps(float(i+1));

    constexpr int INNER = 4096;
    for (int w = 0; w < 3; ++w)
        for (int r = 0; r < INNER; ++r)
            for (auto& a : acc) 
                a = _mm256_fmadd_ps(a, mul, add);
    
    long long reps = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto deadline = t0 + std::chrono::duration<double>(seconds);
    while(std::chrono::steady_clock::now() < deadline) {
        for (int r = 0; r < INNER; ++r)
            for (auto& a : acc)
                a = _mm256_fmadd_ps(a, mul, add);
            reps += INNER;
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    float sink = 0.0f;
    for (auto& a : acc) {
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, a);
        sink += tmp[0];
    }
    asm volatile("" : : "r"(sink) : "memory");

    double flops = double(reps) * NUM_ACC * 8.0 /*floats/vector*/ * 2.0 /*mul+add*/;
    return flops / secs / 1e9;
}

#if defined(__AVX512F__)
template <int NUM_ACC>
double measure_avx512_peak_gflops(double seconds) {
    __m512 acc[NUM_ACC];
    __m512 mul = _mm512_set1_ps(1.0000001f);
    __m512 add = _mm512_set1_ps(1e-6f);
    for (int i = 0; i < NUM_ACC; ++i) 
        acc[i] = _mm512_set1_ps(float(i+1));

    constexpr int INNER = 4096;
    for (int w = 0; w < 3; ++w)
        for (int r = 0; r < INNER; ++r)
            for (auto& a : acc) 
                a = _mm512_fmadd_ps(a, mul, add);
    
    long long reps = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto deadline = t0 + std::chrono::duration<double>(seconds);
    while(std::chrono::steady_clock::now() < deadline) {
        for (int r = 0; r < INNER; ++r)
            for (auto& a : acc)
                a = _mm512_fmadd_ps(a, mul, add);
            reps += INNER;
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    float sink = 0.0f;
    for (auto& a : acc) {
        alignas(64) float tmp[16];
        _mm512_store_ps(tmp, a);
        sink += tmp[0];
    }
    asm volatile("" : : "r"(sink) : "memory");

    double flops = double(reps) * NUM_ACC * 16.0 /*floats/vector*/ * 2.0 /*mul+add*/;
    return flops / secs / 1e9;
}
#endif

double measure_bandwidth_gbps(std::size_t floats_per_array) {
    std::vector<float> a(floats_per_array, 1.0f), b(floats_per_array, 2.0f), c(floats_per_array, 0.0f);
    const float scalar = 3.0f;

    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < floats_per_array; ++i) c[i] = a[i]*scalar + b[i];
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    #ifdef __GNUC__
        asm volatile("" : : "r"(c[0]), "r"(c[floats_per_array - 1]) : "memory");
    #else
        std::atomic_signal_fence(std::memory_order_seq_cst);
    #endif

    double bytes = 3.0 * double(floats_per_array) * sizeof(float);
    return bytes / secs / 1e9;
}

int main() {
    std::puts("=== peak compute (register-resident FMA, no memory traffic) ===");
    double best_avx2 = 0;
    for (int trial = 0; trial < 3; ++trial) {
        double g = measure_avx2_peak_gflops<16>(0.5);
        std::printf("   avx2 (16 acc), trial %d: %6.1f GFLOP/s\n", trial, g);
        best_avx2 = std::max(best_avx2, g);
    }

    double g_acc4 = measure_avx2_peak_gflops<4>(0.5);
    std::printf("   avx2 (4 acc, sanity check): %6.1f GFLOP/s (should be lower than 16-acc)\n", g_acc4);

    #if defined(__AVX512F__) 
        // ... runs same 3-trial test for AVX512 if it exists ...
        double best_avx512 = 0;
        for (int trial = 0; trial < 3; ++trial) {
            double g = measure_avx2_peak_gflops<16>(0.5);
            std::printf("   avx512 (16 acc), trial %d: %6.1f GFLOP/s\n", trial, g);
            best_avx2 = std::max(best_avx2, g);
        }
        std::printf("PEAK_AVX2_GFLOPS=%.1f\nPEAK_AVX512_GFLOPS=%.1f\n", best_avx2, best_avx512);
    #else
        std::printf("PEAK_AVX2_GFLOPS=%.1f\n", best_avx2);
    #endif

    std::puts("\n=== peak memory bandwidth (STREAM-triad-like, 1.5GiB working set, escapes 260MiB L3) ===");
    double best_bw = 0;
    for (int trial = 0; trial < 3; ++trial) {
        double bw = measure_bandwidth_gbps(std::size_t(150) * 1024 * 1024);
        std::printf("  trial %d: %6.1f GB/s\n", trial, bw);
        best_bw = std::max(best_bw, bw);
    }
    std::printf("PEAK_BANDWIDTH_GBPS=%.1f\n", best_bw);

    std::puts("\n=== matmul kernels: (n, GFLOP/s) for the roofline's data points ===");
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int BLOCK = 128;
    for (int n : {128, 256, 512, 1024}) {
        std::vector<float> A(std::size_t(n) * n), B(std::size_t(n) * n), C(std::size_t(n) * n);
        for (auto& v : A) v = dist(rng);
        for (auto& v: B) v = dist(rng);
        double flops = 2.0 * n * n * n;
        
        auto t0 = std::chrono::steady_clock::now();
        tiled(A.data(), B.data(), C.data(), n, BLOCK);
        auto t1 = std::chrono::steady_clock::now();
        double tiled_gflops = flops / std::chrono::duration<double>(t1 - t0).count() / 1e9;

        t0 = std::chrono::steady_clock::now();
        cache_oblivious(A.data(), B.data(), C.data(), n);
        t1 = std::chrono::steady_clock::now();
        double co_gflops = flops / std::chrono::duration<double>(t1 - t0).count() / 1e9;

        // compulsory memroy traffic lower bound: read A, B, write C once each
        double compulsory_bytes = 3.0 * n * n * sizeof(float);
        double ai = flops / compulsory_bytes; // arithmetic intensity
        std::printf("n=%4d  compulsory_AI=%.2f  tiled=%.2f GFLOP/s  cache_oblivious=%.2f GFLOP/s\n", n, ai, tiled_gflops, co_gflops);

        if (n == 256) {
            t0 = std::chrono::steady_clock::now();
            naive_ikj(A.data(), B.data(), C.data(), n);
            t1 = std::chrono::steady_clock::now();
            double ikj_gflops = flops / std::chrono::duration<double>(t1 - t0).count() / 1e9;
            std::printf("  (naive_ikj at n=256, for reference: %.2f GFLOP/s -- true AI is LOWER than "
                        "%.2f, since it re-fetches data the compulsory bound assumes is only "
                        "touched once; not measured exactly here, see README)\n",
                        ikj_gflops, ai);
        }

    }

    return 0;
}


