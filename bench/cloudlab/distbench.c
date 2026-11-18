// Is pgvector's L2 kernel limited by arithmetic or by memory at 1536-d?
//
//   gcc -O2 -march=native -ftree-vectorize -fassociative-math -fno-signed-zeros \
//       -fno-trapping-math -ffp-contract=fast -o distbench distbench.c && ./distbench
//
// Those are pgvector 0.8.7's own flags (its Makefile's PG_CFLAGS, -O2 as in
// Postgres's build). l2_pgvector is VectorL2SquaredDistance from
// pgvector/src/vector.c, verbatim. For each kernel it times one 1536-d distance
// against a fixed query vector:
//   cached     the same base vector every time (both vectors stay in L1/L2)
//   dram       a random base vector from an 8 GiB buffer each time, the HNSW
//              pattern: every element visited is a new 6 KB vector from DRAM
//   dram+pf    as dram, but software-prefetching the next random vector first
// l2_4acc is the same arithmetic with four independent accumulators, to show
// how much of the cached time is the single accumulator's dependency chain.
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DIM 1536
#define VEC_BYTES (DIM * sizeof(float))

__attribute__((noinline)) static float l2_pgvector(int dim, float *ax, float *bx) {
    float distance = 0.0;

    /* Auto-vectorized */
    for (int i = 0; i < dim; i++) {
        float diff = ax[i] - bx[i];

        distance += diff * diff;
    }

    return distance;
}

__attribute__((noinline)) static float l2_4acc(int dim, float *ax, float *bx) {
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    for (int i = 0; i < dim; i += 32) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(ax + i), _mm256_loadu_ps(bx + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(ax + i + 8), _mm256_loadu_ps(bx + i + 8));
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(ax + i + 16), _mm256_loadu_ps(bx + i + 16));
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(ax + i + 24), _mm256_loadu_ps(bx + i + 24));
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        s1 = _mm256_fmadd_ps(d1, d1, s1);
        s2 = _mm256_fmadd_ps(d2, d2, s2);
        s3 = _mm256_fmadd_ps(d3, d3, s3);
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    float lanes[8];
    _mm256_storeu_ps(lanes, s);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

typedef float (*kernel_t)(int, float *, float *);

static uint64_t rng = 0x2545F4914F6CDD1DULL;
static inline uint64_t next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static double bench(kernel_t k, float *q, float *base, size_t nvec, int mode, volatile float *sink) {
    const int calls = mode == 0 ? 2000000 : 400000;
    size_t cur = next() % nvec;
    float acc = 0;
    double t0 = now();
    for (int i = 0; i < calls; i++) {
        size_t nxt = next() % nvec;
        if (mode == 2) {  // prefetch every cache line of the next vector
            const char *p = (const char *)(base + nxt * DIM);
            for (size_t o = 0; o < VEC_BYTES; o += 64) _mm_prefetch(p + o, _MM_HINT_T0);
        }
        acc += k(DIM, q, mode == 0 ? base : base + cur * DIM);
        cur = nxt;
    }
    double ns = (now() - t0) / calls * 1e9;
    *sink = acc;
    return ns;
}

int main(void) {
    const size_t nvec = (8ULL << 30) / VEC_BYTES;
    float *base, *q;
    if (posix_memalign((void **)&base, 64, nvec * VEC_BYTES) || posix_memalign((void **)&q, 64, VEC_BYTES)) return 1;
    for (size_t i = 0; i < nvec * DIM; i++) base[i] = (float)(i % 97) * 0.01f;
    for (int i = 0; i < DIM; i++) q[i] = (float)(i % 89) * 0.01f;
    volatile float sink;
    const char *names[] = {"cached", "dram", "dram+pf"};
    printf("%-12s %10s %10s %10s   (ns per 1536-d distance, 1 thread)\n", "kernel", names[0], names[1], names[2]);
    struct { const char *n; kernel_t k; } ks[] = {{"l2_pgvector", l2_pgvector}, {"l2_4acc", l2_4acc}};
    for (int j = 0; j < 2; j++) {
        printf("%-12s", ks[j].n);
        for (int mode = 0; mode < 3; mode++) printf(" %10.0f", bench(ks[j].k, q, base, nvec, mode, &sink));
        printf("\n");
    }
    return 0;
}
