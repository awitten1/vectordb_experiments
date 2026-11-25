/*
 * Time per call of pgvectorfast_ntq's 4+8 bit rerank distance (HnswTqDistance4k
 * at 8 extra bits) with each kernel built: scalar, AVX2, AVX-512. Encoded
 * points at dimension 384 (Hacker News MiniLM), warm in cache, so this is the
 * kernel's own cost, without the reads a rerank also does.
 *
 *   cc -O2 -ffp-contract=off -march=native -Ipgvectorfast_ntq/src \
 *     pgvectorfast_ntq/src/hnswtq.c bench/cloudlab/tq_rerank_bench.c -lm -o tq_rerank_bench
 *   ./tq_rerank_bench [dim]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "hnswtq.h"

#define POINTS 1024
#define REPS 200

static uint64_t rng = 88172645463325252ULL;

static float
Normal(void)
{
	double		s = 0.0;

	for (int i = 0; i < 12; i++)
	{
		rng ^= rng >> 12;
		rng ^= rng << 25;
		rng ^= rng >> 27;
		s += (double) ((rng * 2685821657736338717ULL) >> 11) / 9007199254740992.0;
	}
	return (float) (s - 6.0);
}

static double
Now(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int
main(int argc, char **argv)
{
	const int	dim = argc > 1 ? atoi(argv[1]) : 384;
	static const int kernels[] = {HNSW_TQ_KERNEL_SCALAR, HNSW_TQ_KERNEL_AVX2, HNSW_TQ_KERNEL_AVX512};
	static const char *names[] = {"scalar", "avx2", "avx512"};
	const size_t cb = HnswTqCodeBytes(dim), eb = HnswTqExtraBytes(dim, 8);
	HnswTqCodec *codec = malloc(HnswTqCodecSize(dim));
	float	   *scratch = malloc(HnswTqScratchSize(dim));
	HnswTqQuery *q = HnswTqInitQuery(malloc(HnswTqQuerySize(dim) + 8), dim);
	uint8_t    *codes = malloc(cb * POINTS);
	uint8_t    *extra = malloc(eb * POINTS);
	HnswTqPointInfo *info = malloc(sizeof(HnswTqPointInfo) * POINTS);
	float	   *x = malloc(sizeof(float) * dim);
	double		base = 0.0;

	if (!HnswTqInitCodec(codec, dim, HNSW_TQ_DEFAULT_SEED, NULL))
	{
		fprintf(stderr, "unsupported dimension %d\n", dim);
		return 1;
	}
	for (int p = 0; p < POINTS; p++)
	{
		for (int i = 0; i < dim; i++)
			x[i] = Normal();
		HnswTqEncodePoint(codec, x, 8, codes + p * cb, extra + p * eb, &info[p], scratch);
	}
	for (int i = 0; i < dim; i++)
		x[i] = Normal();
	HnswTqEncodeQuery(codec, x, q, scratch);

	for (int k = 0; k < 3; k++)
	{
		volatile float sink = 0.0f;
		double		t0,
					ns;

		if (!HnswTqKernelAvailable(kernels[k]))
		{
			printf("%-7s not built\n", names[k]);
			continue;
		}
		hnsw_tq_rerank_kernel = kernels[k];
		for (int p = 0; p < POINTS; p++)	/* warm up */
			sink += HnswTqDistance4k(q, codes + p * cb, extra + p * eb, 8, &info[p], HNSW_TQ_METRIC_L2);
		t0 = Now();
		for (int r = 0; r < REPS; r++)
			for (int p = 0; p < POINTS; p++)
				sink += HnswTqDistance4k(q, codes + p * cb, extra + p * eb, 8, &info[p], HNSW_TQ_METRIC_L2);
		ns = (Now() - t0) * 1e9 / ((double) REPS * POINTS);
		if (k == 0)
			base = ns;
		printf("%-7s %8.1f ns per distance (%.1fx the scalar loop's speed), dim %d\n", names[k], ns, base / ns, dim);
	}
	return 0;
}
