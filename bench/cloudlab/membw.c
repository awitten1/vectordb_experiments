// Memory ceilings of this machine, for reading the HNSW profiles against.
//
//   gcc -O2 -march=native -pthread -o membw membw.c && ./membw [GiB]
//
// For 1, 2, 4, 8, 12 and 16 threads:
//   seq     each thread streams through its own slice of the buffer
//   chunk6k each thread reads whole random 6144-byte chunks: one 1536-d float
//           vector per HNSW element visited, which is pgvector's access pattern
// Then a single-thread pointer chase over random cache lines: DRAM latency.
// Results are GB/s of data actually read (1 GB = 1e9 bytes) and ns per load.
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHUNK 6144
static size_t total_bytes;
static uint8_t *buf;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

typedef struct {
    int id, nthreads, mode;
    double seconds;
    size_t bytes;
    uint64_t sink;
} job_t;

static pthread_barrier_t barrier;

static void *worker(void *arg) {
    job_t *j = arg;
    __m256i acc = _mm256_setzero_si256();
    uint64_t rng = 0x9e3779b97f4a7c15ULL * (j->id + 1);
    size_t nchunks = total_bytes / CHUNK, bytes = 0;
    size_t slice = total_bytes / j->nthreads, lo = slice * j->id;
    pthread_barrier_wait(&barrier);
    double t0 = now(), t;
    do {
        if (j->mode == 0) {  // one sequential pass over this thread's slice
            for (size_t o = lo; o + 32 <= lo + slice; o += 32)
                acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i *)(buf + o)));
            bytes += slice;
        } else {  // 4096 random whole chunks
            for (int r = 0; r < 4096; r++) {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                const uint8_t *c = buf + (rng % nchunks) * CHUNK;
                for (int o = 0; o < CHUNK; o += 32)
                    acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i *)(c + o)));
            }
            bytes += (size_t)4096 * CHUNK;
        }
        t = now();
    } while (t - t0 < 2.0);
    uint64_t lanes[4];
    _mm256_storeu_si256((__m256i *)lanes, acc);
    j->sink = lanes[0] ^ lanes[1] ^ lanes[2] ^ lanes[3];
    j->seconds = t - t0;
    j->bytes = bytes;
    return NULL;
}

static double run(int nthreads, int mode) {
    pthread_t th[64];
    job_t jobs[64];
    pthread_barrier_init(&barrier, NULL, nthreads);
    for (int i = 0; i < nthreads; i++) {
        jobs[i] = (job_t){.id = i, .nthreads = nthreads, .mode = mode};
        pthread_create(&th[i], NULL, worker, &jobs[i]);
    }
    double gbps = 0;
    for (int i = 0; i < nthreads; i++) {
        pthread_join(th[i], NULL);
        gbps += jobs[i].bytes / jobs[i].seconds / 1e9;
    }
    pthread_barrier_destroy(&barrier);
    return gbps;
}

int main(int argc, char **argv) {
    double gib = argc > 1 ? atof(argv[1]) : 8.0;
    total_bytes = ((size_t)(gib * (1ULL << 30)) / CHUNK) * CHUNK;
    if (posix_memalign((void **)&buf, 2 << 20, total_bytes)) { perror("alloc"); return 1; }
    memset(buf, 1, total_bytes);  // fault every page in before timing

    printf("buffer %.1f GiB\n%8s %10s %12s\n", total_bytes / (double)(1ULL << 30), "threads", "seq GB/s", "chunk6k GB/s");
    int counts[] = {1, 2, 4, 8, 12, 16};
    for (size_t i = 0; i < sizeof counts / sizeof *counts; i++)
        printf("%8d %10.1f %12.1f\n", counts[i], run(counts[i], 0), run(counts[i], 1));

    // Pointer chase: a random cyclic permutation of cache lines across the buffer.
    size_t lines = total_bytes / 64;
    size_t *perm = malloc(lines * sizeof *perm);
    for (size_t i = 0; i < lines; i++) perm[i] = i;
    uint64_t rng = 88172645463325252ULL;
    for (size_t i = lines - 1; i > 0; i--) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        size_t k = rng % (i + 1), tmp = perm[i]; perm[i] = perm[k]; perm[k] = tmp;
    }
    for (size_t i = 0; i < lines; i++)
        *(uint64_t *)(buf + perm[i] * 64) = perm[(i + 1) % lines] * 64;
    free(perm);
    uint64_t p = 0;
    const size_t steps = 20000000;
    double t0 = now();
    for (size_t i = 0; i < steps; i++) p = *(volatile uint64_t *)(buf + p);
    double ns = (now() - t0) / steps * 1e9;
    printf("pointer-chase latency: %.1f ns per dependent load (sink %llu)\n", ns, (unsigned long long)(p & 1));
    return 0;
}
