/*
 * Distance-kernel microbenchmark: pgvector's float L2 vs pgvector_ntq's tq4
 * navigation dot product (TqNavDot), at 384 dimensions, with no Postgres in
 * the way.
 *
 *   gcc -O2 -march=native -ftree-vectorize -fassociative-math -fno-signed-zeros \
 *       -fno-trapping-math -ffp-contract=fast -o tq_kernel_bench tq_kernel_bench.c -lm
 *   ./tq_kernel_bench [rows=8000000] [lookups=4000000]
 *
 * The flags are pgvector's own (Makefile PG_CFLAGS), so the loops compile as
 * they do in the extensions. The two kernels are copied verbatim:
 *   l2f      pgvector/src/vector.c VectorL2SquaredDistance (1536 B per row)
 *   navdot   pgvector_ntq/src/hnswtq.c TqNavDot (4-bit codes, 192 B per row
 *            at pdim 384 = 3 x 128, the padding TqChoosePadding picks)
 *
 * Cases, each reported as ns per distance:
 *   in-cache  64 rows, reused: pure compute
 *   indep     random rows over `rows`, indices known up front: memory-level
 *             parallelism allowed (the upper bound on overlap)
 *   batch16   16 independent random rows, then the next 16 depend on their
 *             results: roughly an HNSW expansion (one neighbor list at a time)
 *   chain     every row depends on the previous result: fully serial misses
 * Each figure is the better of two passes. The random cases run with 4 KB pages (MADV_NOHUGEPAGE) and with transparent
 * huge pages (MADV_HUGEPAGE), whose AnonHugePages share is printed.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define DIM 384
#define PDIM 384
#define CODE_BYTES (PDIM / 2)
#define FROW (DIM * sizeof(float))

static const float hnsw_tq4_centroids[8] = {
	1.283950299e-01f, 3.880482995e-01f, 6.567591185e-01f, 9.423404565e-01f,
	1.256231197e+00f, 1.618046386e+00f, 2.069017227e+00f, 2.732589571e+00f
};
static float tq4_lut[16];

/* pgvector/src/vector.c */
__attribute__((noinline)) static float
VectorL2SquaredDistance(int dim, float *ax, float *bx)
{
	float		distance = 0.0;

	/* Auto-vectorized */
	for (int i = 0; i < dim; i++)
	{
		float		diff = ax[i] - bx[i];

		distance += diff * diff;
	}

	return distance;
}

/* pgvector_ntq/src/hnswtq.c */
__attribute__((noinline)) static float
TqNavDot(const float *qr, const uint8_t *code, int pdim)
{
	float		acc0 = 0;
	float		acc1 = 0;
	int			half = pdim / 2;

	for (int i = 0; i < half; i++)
	{
		uint8_t		b = code[i];

		acc0 += qr[2 * i] * tq4_lut[b & 0x0F];
		acc1 += qr[2 * i + 1] * tq4_lut[b >> 4];
	}

	if (pdim & 1)
		acc0 += qr[pdim - 1] * tq4_lut[code[half] & 0x0F];

	return acc0 + acc1;
}

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static inline uint64_t
next64(void)
{
	uint64_t	z = (rng += 0x9E3779B97F4A7C15ull);

	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	return z ^ (z >> 31);
}

static double
now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static void *
alloc(size_t bytes, int huge)
{
	void	   *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

	if (p == MAP_FAILED)
	{
		perror("mmap");
		exit(1);
	}
#ifdef MADV_HUGEPAGE
	if (madvise(p, bytes, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE) != 0)
		perror("madvise");
#endif
	return p;
}

static long
anon_huge_kb(void)
{
	FILE	   *f = fopen("/proc/self/smaps_rollup", "r");
	char		line[256];
	long		kb = -1;

	if (!f)
		return -1;
	while (fgets(line, sizeof line, f))
		if (sscanf(line, "AnonHugePages: %ld kB", &kb) == 1)
			break;
	fclose(f);
	return kb;
}

/*
 * The next row: a fresh random index that cannot be loaded before the
 * previous distance is known. dep() is 0 for any finite distance, but the
 * address still waits on it (x86 does not predict values). Iterating a
 * function of the distance alone would fall into a short cycle that fits in
 * cache.
 */
static inline size_t
dep(float d)
{
	uint32_t	bits;

	memcpy(&bits, &d, 4);
	return (bits >> 31) & ((bits >> 23) == 0xFF);
}

typedef float (*dist_fn) (const float *q, const void *base, size_t row);

static float
f_l2(const float *q, const void *base, size_t row)
{
	return VectorL2SquaredDistance(DIM, (float *) q, (float *) ((const char *) base + row * FROW));
}

static float
f_nav(const float *q, const void *base, size_t row)
{
	return TqNavDot(q, (const uint8_t *) base + row * CODE_BYTES, PDIM);
}

static volatile float sink;

static double run1(dist_fn f, const float *q, const void *base, size_t rows, const uint32_t *idx, size_t lookups, int mode);

/* Best of two passes: the first can include page-fault or reclaim noise */
static double
run(dist_fn f, const float *q, const void *base, size_t rows, const uint32_t *idx, size_t lookups, int mode)
{
	double		a = run1(f, q, base, rows, idx, lookups, mode);
	double		b = run1(f, q, base, rows, idx, lookups, mode);

	return a < b ? a : b;
}

static double
run1(dist_fn f, const float *q, const void *base, size_t rows, const uint32_t *idx, size_t lookups, int mode)
{
	float		acc = 0;
	double		t0 = now_ns();

	if (mode == 0)				/* indep */
		for (size_t i = 0; i < lookups; i++)
			acc += f(q, base, idx[i]);
	else if (mode == 1)			/* batch16 */
	{
		size_t		seed = 0;

		for (size_t i = 0; i < lookups; i += 16)
		{
			float		best = INFINITY;

			for (int j = 0; j < 16; j++)
			{
				float		d = f(q, base, (seed + idx[i + j]) % rows);

				best = d < best ? d : best;
			}
			acc += best;
			seed = dep(best);
		}
	}
	else						/* chain */
	{
		size_t		row = idx[0];

		for (size_t i = 1; i <= lookups; i++)
		{
			float		d = f(q, base, row);

			acc += d;
			row = (idx[i % lookups] + dep(d)) % rows;
		}
	}
	sink = acc;
	return (now_ns() - t0) / lookups;
}

static void
fill(float *fl, uint8_t *codes, size_t rows)
{
	for (size_t r = 0; r < rows; r++)
	{
		for (int i = 0; i < DIM; i++)
			fl[r * DIM + i] = (float) ((int64_t) (next64() >> 40) - (1 << 23)) / (1 << 23) * 0.1f;
		for (int i = 0; i < CODE_BYTES; i += 8)
		{
			uint64_t	v = next64();

			memcpy(codes + r * CODE_BYTES + i, &v, 8);
		}
	}
}

static void
check(const float *q, const float *fl, const uint8_t *codes)
{
	double		l2 = 0,
				nav = 0;

	for (int i = 0; i < DIM; i++)
	{
		double		d = (double) q[i] - fl[i];

		l2 += d * d;
	}
	for (int i = 0; i < PDIM; i++)
	{
		int			nib = (codes[i >> 1] >> ((i & 1) << 2)) & 0x0F;

		nav += (double) q[i] * tq4_lut[nib];
	}
	float		a = f_l2(q, fl, 0),
				b = f_nav(q, codes, 0);

	if (fabs(a - l2) > 1e-3 * fabs(l2) + 1e-4 || fabs(b - nav) > 1e-3 * fabs(nav) + 1e-4)
	{
		fprintf(stderr, "error: kernel mismatch: l2 %g vs %g, navdot %g vs %g\n", a, l2, b, nav);
		exit(1);
	}
	printf("reference check ok: l2 %.6f (ref %.6f), navdot %.6f (ref %.6f)\n", a, l2, b, nav);
}

int
main(int argc, char **argv)
{
	size_t		rows = argc > 1 ? strtoull(argv[1], NULL, 10) : 8000000;
	size_t		lookups = argc > 2 ? strtoull(argv[2], NULL, 10) : 4000000;
	float		q[DIM];
	uint32_t   *idx = malloc(lookups * sizeof(uint32_t));
	const char *modes[] = {"indep", "batch16", "chain"};

	lookups -= lookups % 16;
	for (int n = 0; n < 16; n++)
		tq4_lut[n] = (n & 8 ? -1.0f : 1.0f) * hnsw_tq4_centroids[n & 7];
	for (int i = 0; i < DIM; i++)
		q[i] = (float) ((int64_t) (next64() >> 40) - (1 << 23)) / (1 << 23) * 0.1f;

	printf("rows %zu, lookups %zu: floats %.1f GB (%zu B/row), codes %.2f GB (%d B/row)\n",
		   rows, lookups, rows * FROW / 1e9, FROW, rows * (double) CODE_BYTES / 1e9, CODE_BYTES);

	/* In-cache: 64 rows */
	{
		float	   *fl = aligned_alloc(64, 64 * FROW);
		uint8_t    *codes = aligned_alloc(64, 64 * CODE_BYTES);
		uint32_t   *small = malloc(lookups * sizeof(uint32_t));

		fill(fl, codes, 64);
		check(q, fl, codes);
		for (size_t i = 0; i < lookups; i++)
			small[i] = next64() % 64;
		run(f_l2, q, fl, 64, small, lookups, 0);	/* warm */
		run(f_nav, q, codes, 64, small, lookups, 0);
		printf("%-10s %-8s %-6s %8.1f ns/distance\n", "in-cache", "l2f", "-", run(f_l2, q, fl, 64, small, lookups, 0));
		printf("%-10s %-8s %-6s %8.1f ns/distance\n", "in-cache", "navdot", "-", run(f_nav, q, codes, 64, small, lookups, 0));
		free(fl);
		free(codes);
		free(small);
	}

	for (size_t i = 0; i < lookups; i++)
		idx[i] = next64() % rows;

	for (int huge = 0; huge <= 1; huge++)
	{
		float	   *fl = alloc(rows * FROW, huge);
		uint8_t    *codes = alloc(rows * CODE_BYTES, huge);

		fill(fl, codes, rows);
		printf("pages: %s (AnonHugePages %ld MB)\n", huge ? "THP" : "4 KB", anon_huge_kb() / 1024);
		for (int m = 0; m < 3; m++)
		{
			double		a = run(f_l2, q, fl, rows, idx, lookups, m);
			double		b = run(f_nav, q, codes, rows, idx, lookups, m);

			printf("%-10s %-8s %-6s %8.1f ns/distance\n", modes[m], "l2f", huge ? "THP" : "4KB", a);
			printf("%-10s %-8s %-6s %8.1f ns/distance   (navdot/l2f %.2fx)\n", modes[m], "navdot", huge ? "THP" : "4KB", b, b / a);
		}
		munmap(fl, rows * FROW);
		munmap(codes, rows * CODE_BYTES);
	}
	return 0;
}
