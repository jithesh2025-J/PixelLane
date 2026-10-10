
#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>

using namespace std;
using namespace chrono;

const int N = 1000000;

// Scalar processing: one integer at a time
void scalarProcess(const int* a, const int* b, int* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// AVX2 SIMD processing: eight 32-bit integers at a time
void simdProcess(const int* a, const int* b, int* c, int n) {
    int i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(&a[i]));

        __m256i vb = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(&b[i]));

        __m256i vc = _mm256_add_epi32(va, vb);

        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(&c[i]), vc);
    }

    // Handle remaining elements
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    vector<int> a(N), b(N);
    vector<int> scalarResult(N), simdResult(N);

    // Generate sample data
    for (int i = 0; i < N; i++) {
        a[i] = i % 100;
        b[i] = (i * 2) % 100;
    }

    // Measure scalar execution time
    auto start = high_resolution_clock::now();

    scalarProcess(a.data(), b.data(), scalarResult.data(), N);

    auto end = high_resolution_clock::now();

    double scalarTime =
        duration<double, milli>(end - start).count();

    // Measure AVX2 SIMD execution time
    start = high_resolution_clock::now();

    simdProcess(a.data(), b.data(), simdResult.data(), N);

    end = high_resolution_clock::now();

    double simdTime =
        duration<double, milli>(end - start).count();

    // Verify correctness
    bool correct = (scalarResult == simdResult);

    // Display results
    cout << "SIMD-Based Array Processing Accelerator\n";
    cout << "========================================\n";
    cout << "CPU: Intel Core i5-10300H\n";
    cout << "Array size: " << N << " integers\n\n";

    cout << "Scalar execution time: "
         << scalarTime << " ms\n";

    cout << "AVX2 SIMD execution time: "
         << simdTime << " ms\n";

    if (simdTime > 0.0) {
        cout << "Speedup: "
             << scalarTime / simdTime << "x\n";
    }

    cout << "Results match: "
         << (correct ? "YES" : "NO") << "\n";

    cout << "Sample result at index 100: "
         << simdResult[100] << "\n";

    return correct ? 0 : 1;
}
