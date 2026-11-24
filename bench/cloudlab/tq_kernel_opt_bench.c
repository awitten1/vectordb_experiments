/*
 * Correctness and speed of the fast TurboQuant kernels (pgvectorfast_ntq's
 * src/hnswtq_simd.h, included as is) against the original loops from
 * pgvector_ntq/src/hnswtq.c (copied verbatim below).
 *
 *   gcc -O2 -march=native -ftree-vectorize -fassociative-math -fno-signed-zeros \
 *       -fno-trapping-math -ffp-contract=fast -I pgvectorfast_ntq/src \
 *       -o tq_kernel_opt_bench bench/cloudlab/tq_kernel_opt_bench.c -lm
 *
 * Same flags as the extensions (their Makefile PG_CFLAGS). For each kernel and
 * pdim (384 = the HN index, 1536, and sizes that exercise the tails), checks
 * the original and the fast kernel against a double-precision reference on
 * random inputs, then times both on 64 in-cache rows: ns per call.
 * Exits non-zero if any result is off by more than 1e-5 of the sum of
 * absolute terms.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hnswtq_tables.h"
#include "hnswtq_simd.h"

typedef uint8_t uint8;
typedef uint32_t uint32;

static float tq4_lut[16];

/* ---- original kernels, verbatim from pgvector_ntq/src/hnswtq.c ---- */
static inline int
TqNibble(const uint8 *code, int i)
{
	return (code[i >> 1] >> ((i & 1) << 2)) & 0x0F;
}

__attribute__((noinline)) static float
TqNavDot(const float *qr, const uint8 *code, int pdim)
{
	float		acc0 = 0;
	float		acc1 = 0;
	int			half = pdim / 2;

	for (int i = 0; i < half; i++)
	{
		uint8		b = code[i];

		acc0 += qr[2 * i] * tq4_lut[b & 0x0F];
		acc1 += qr[2 * i + 1] * tq4_lut[b >> 4];
	}

	if (pdim & 1)
		acc0 += qr[pdim - 1] * tq4_lut[code[half] & 0x0F];

	return acc0 + acc1;
}

static inline uint32
TqGetBits(const uint8 *buf, int i, int k)
{
	uint32		off;
	uint32		w;

	if (k == 8)
		return buf[i];

	off = (uint32) i * k;
	w = (uint32) buf[off >> 3] | ((uint32) buf[(off >> 3) + 1] << 8);
	return (w >> (off & 7)) & ((1U << k) - 1);
}

static inline const float *
TqRefineCells(int k, int u)
{
	return k == 6 ? hnsw_tq_ref6[u] : hnsw_tq_ref8[u];
}

__attribute__((noinline)) static float
TqRefineDot(const float *qr, const uint8 *code, const uint8 *ref, int pdim, int k)
{
	float		acc = 0;

	for (int i = 0; i < pdim; i++)
	{
		int			nib = TqNibble(code, i);
		float		c = TqRefineCells(k, nib & 7)[TqGetBits(ref, i, k)];

		acc += qr[i] * (nib & 8 ? -c : c);
	}

	return acc;
}

/* ---- the fast kernels, behind a call boundary like the originals ---- */
__attribute__((noinline)) static float
NavFast(const float *qr, const uint8 *code, int pdim)
{
	return TqNavDotFast(qr, code, pdim, hnsw_tq4_centroids, tq4_lut);
}

__attribute__((noinline)) static float
RefineFast(const float *qr, const uint8 *code, const uint8 *ref, int pdim, int k)
{
	return TqRefineDotFast(qr, code, ref, pdim, k, k == 6 ? &hnsw_tq_ref6[0][0] : &hnsw_tq_ref8[0][0]);
}

/* ---- reference in double, and its scale ---- */
static double
RefNav(const float *q, const uint8 *code, int pdim, double *l1)
{
	double		s = 0;

	*l1 = 0;
	for (int i = 0; i < pdim; i++)
	{
		double		t = (double) q[i] * tq4_lut[TqNibble(code, i)];

		s += t;
		*l1 += fabs(t);
	}
	return s;
}

static double
RefRefine(const float *q, const uint8 *code, const uint8 *ref, int pdim, int k, double *l1)
{
	double		s = 0;

	*l1 = 0;
	for (int i = 0; i < pdim; i++)
	{
		int			nib = TqNibble(code, i);
		double		c = TqRefineCells(k, nib & 7)[TqGetBits(ref, i, k)];
		double		t = (double) q[i] * (nib & 8 ? -c : c);

		s += t;
		*l1 += fabs(t);
	}
	return s;
}

static uint64_t rng = 0x243F6A8885A308D3ull;
static uint64_t
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

#define ROWS 64
#define MAXP 1536
#define CALLS 2000000

static volatile float sink;

int
main(void)
{
	static float q[ROWS][MAXP];
	static uint8 code[ROWS][MAXP / 2 + 8];
	static uint8 ref[ROWS][MAXP + 8];
	const int	pdims[] = {384, 1536, 120, 13, 1};
	int			failures = 0;

	for (int n = 0; n < 16; n++)
		tq4_lut[n] = (n & 8 ? -1.0f : 1.0f) * hnsw_tq4_centroids[n & 7];
	for (int r = 0; r < ROWS; r++)
	{
		for (int i = 0; i < MAXP; i++)
			q[r][i] = (float) ((int64_t) (next64() >> 40) - (1 << 23)) / (1 << 23);
		for (int i = 0; i < MAXP / 2 + 8; i++)
			code[r][i] = (uint8) next64();
		for (int i = 0; i < MAXP + 8; i++)
			ref[r][i] = (uint8) next64();
	}

#ifdef HNSW_TQ_AVX2
	printf("fast kernels: AVX2 path\n");
#else
	printf("fast kernels: scalar path (no AVX2)\n");
#endif

	/* correctness */
	for (size_t p = 0; p < sizeof(pdims) / sizeof(pdims[0]); p++)
	{
		int			pdim = pdims[p];
		double		worst[5] = {0};

		for (int t = 0; t < 2000; t++)
		{
			int			r = t % ROWS;
			double		l1,
						ref_v;

			/* vary the inputs per trial */
			for (int i = 0; i < pdim / 2 + 1; i++)
				code[r][i] = (uint8) next64();
			for (int i = 0; i < pdim + 1; i++)
				ref[r][i] = (uint8) next64();

			ref_v = RefNav(q[r], code[r], pdim, &l1);
			worst[0] = fmax(worst[0], fabs(TqNavDot(q[r], code[r], pdim) - ref_v) / (l1 + 1e-30));
			worst[1] = fmax(worst[1], fabs(NavFast(q[r], code[r], pdim) - ref_v) / (l1 + 1e-30));
			for (int kk = 0; kk < 2; kk++)
			{
				int			k = kk ? 6 : 8;

				ref_v = RefRefine(q[r], code[r], ref[r], pdim, k, &l1);
				worst[2 + kk] = fmax(worst[2 + kk], fabs(RefineFast(q[r], code[r], ref[r], pdim, k) - ref_v) / (l1 + 1e-30));
				worst[4] = fmax(worst[4], fabs(TqRefineDot(q[r], code[r], ref[r], pdim, k) - ref_v) / (l1 + 1e-30));
			}
		}
		printf("pdim %4d  max rel error: nav orig %.1e fast %.1e | refine k8 fast %.1e k6 fast %.1e orig %.1e\n",
			   pdim, worst[0], worst[1], worst[2], worst[3], worst[4]);
		for (int j = 0; j < 5; j++)
			if (!(worst[j] <= 1e-5))
				failures++;
	}

	/* speed: in-cache, 64 rows, pdim 384 */
	{
		const int	pdim = 384;
		double		t0,
					t;
		float		acc = 0;

#define TIME(label, expr) \
		do { \
			for (int w = 0; w < 2; w++) { \
				acc = 0; t0 = now_ns(); \
				for (int c = 0; c < CALLS; c++) { int r = c & (ROWS - 1); acc += (expr); } \
				t = (now_ns() - t0) / CALLS; sink = acc; \
			} \
			printf("%-28s %7.1f ns/call\n", label, t); \
		} while (0)

		TIME("navdot  original", TqNavDot(q[r], code[r], pdim));
		TIME("navdot  fast", NavFast(q[r], code[r], pdim));
		TIME("refine8 original", TqRefineDot(q[r], code[r], ref[r], pdim, 8));
		TIME("refine8 fast", RefineFast(q[r], code[r], ref[r], pdim, 8));
		TIME("refine6 original", TqRefineDot(q[r], code[r], ref[r], pdim, 6));
		TIME("refine6 fast", RefineFast(q[r], code[r], ref[r], pdim, 6));
	}

	if (failures)
	{
		fprintf(stderr, "error: %d kernel/size combinations exceed 1e-5 relative error\n", failures);
		return 1;
	}
	printf("all kernels within 1e-5 of the double reference\n");
	return 0;
}
