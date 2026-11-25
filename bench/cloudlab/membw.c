// Memory bandwidth and latency on one node, for the HN study's bottleneck notes.
//   gcc -O3 -march=native -ffast-math -fopenmp membw.c -o membw  (-ffast-math lets the sum vectorise)
//   numactl --interleave=all ./membw        (as Postgres ran: memory on both sockets)
// read:  every thread sums its slice of a 16 GB array (DRAM read bandwidth)
// triad: a[i] = b[i] + s * c[i] over 3 x 4 GB (STREAM's triad; counts 3 x 8 B per element)
// chase: one thread follows a random cycle through 4 GB (load-to-use latency of a
//        dependent DRAM access, the pattern of a graph walk without prefetch)
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static double read_bw(const double *a, size_t n, int threads) {
  double best = 0, sink = 0;
  for (int rep = 0; rep < 5; rep++) {
    double t0 = now(), s = 0;
#pragma omp parallel for num_threads(threads) reduction(+ : s) schedule(static)
    for (size_t i = 0; i < n; i++) s += a[i];
    double t = now() - t0;
    sink += s;
    double gbs = n * 8.0 / t / 1e9;
    if (gbs > best) best = gbs;
  }
  if (sink == 42) puts("");
  return best;
}

static double triad_bw(double *a, const double *b, const double *c, size_t n, int threads) {
  double best = 0;
  for (int rep = 0; rep < 5; rep++) {
    double t0 = now();
#pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t i = 0; i < n; i++) a[i] = b[i] + 3.0 * c[i];
    double t = now() - t0;
    double gbs = n * 24.0 / t / 1e9;
    if (gbs > best) best = gbs;
  }
  return best;
}

int main(void) {
  int maxt = omp_get_max_threads();
  size_t n = (size_t)2 << 30;  // 2 Gi doubles = 16 GB
  double *a = aligned_alloc(64, n * 8);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++) a[i] = 1.0;
  printf("read  %2d threads: %6.1f GB/s\n", maxt, read_bw(a, n, maxt));
  printf("read  %2d threads: %6.1f GB/s\n", maxt / 2, read_bw(a, n, maxt / 2));
  printf("read   1 thread:  %6.1f GB/s\n", read_bw(a, n / 8, 1));
  free(a);

  size_t m = (size_t)512 << 20;  // 512 Mi doubles = 4 GB per array
  double *x = aligned_alloc(64, m * 8), *y = aligned_alloc(64, m * 8), *z = aligned_alloc(64, m * 8);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < m; i++) { x[i] = 0; y[i] = 1; z[i] = 2; }
  printf("triad %2d threads: %6.1f GB/s\n", maxt, triad_bw(x, y, z, m, maxt));
  free(x); free(y); free(z);

  // Random cyclic permutation over 4 GB of cache-line-sized slots (Sattolo).
  size_t slots = ((size_t)4 << 30) / 64;
  uint64_t *p = aligned_alloc(64, slots * 64);
  size_t *perm = malloc(slots * sizeof(size_t));
  for (size_t i = 0; i < slots; i++) perm[i] = i;
  uint64_t r = 88172645463325252ull;
  for (size_t i = slots - 1; i > 0; i--) {
    r ^= r << 13; r ^= r >> 7; r ^= r << 17;
    size_t j = r % i;
    size_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }
  for (size_t i = 0; i < slots; i++) p[perm[i] * 8] = perm[(i + 1) % slots] * 8;
  free(perm);
  size_t steps = 20000000, cur = 0;
  double t0 = now();
  for (size_t i = 0; i < steps; i++) cur = p[cur];
  double t = now() - t0;
  printf("chase  1 thread:  %6.1f ns per dependent access (%zu)\n", t / steps * 1e9, cur % 2);
  free(p);
  return 0;
}
