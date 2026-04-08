/* bench.c — LCVDB library benchmark.
 *
 * Measures insert throughput, search latency, and save/load speed
 * at various scales.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "lcvdb.h"

static double get_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static float *gen_vectors(int n, int dim, uint32_t seed) {
    float *v = malloc((size_t)n * dim * sizeof(float));
    for (int i = 0; i < n * dim; i++) {
        seed = seed * 1103515245 + 12345;
        v[i] = ((float)(seed >> 16) / 32768.0f) - 1.0f;
    }
    return v;
}

static void bench_scale(int n, int dim) {
    printf("--- N=%dk, dim=%d ---\n", n / 1000, dim);

    float *vectors = gen_vectors(n, dim, 42);
    float *queries = gen_vectors(200, dim, 999);

    /* Insert */
    lcvdb_t *db = lcvdb_create((uint16_t)dim, 1024);
    if (!db) { printf("FAIL: create\n"); free(vectors); free(queries); return; }

    double t0 = get_ns();
    for (int i = 0; i < n; i++) {
        lcvdb_insert(db, (uint32_t)i, vectors + (size_t)i * dim);
    }
    double insert_ns = get_ns() - t0;
    printf("  Insert:  %.1f ms total  (%.0f vec/s)\n",
           insert_ns / 1e6, n / (insert_ns / 1e9));

    /* Search */
    uint32_t ids[10];
    float scores[10];
    t0 = get_ns();
    for (int q = 0; q < 200; q++) {
        lcvdb_search(db, queries + (size_t)q * dim, 10, ids, scores);
    }
    double search_ns = get_ns() - t0;
    printf("  Search:  %.3f ms/query  (%.0f QPS)\n",
           search_ns / 200 / 1e6, 200 / (search_ns / 1e9));

    /* Self-search R@1 */
    int r1 = 0;
    for (int q = 0; q < 200 && q < n; q++) {
        int found = lcvdb_search(db, vectors + (size_t)q * dim, 1, ids, scores);
        if (found > 0 && ids[0] == (uint32_t)q) r1++;
    }
    int nq_r1 = (200 < n) ? 200 : n;
    printf("  R@1:     %.1f%% (%d/%d self-search)\n",
           r1 * 100.0 / nq_r1, r1, nq_r1);

    /* Save */
    t0 = get_ns();
    lcvdb_save(db, "/tmp/lcvdb_bench.db");
    double save_ns = get_ns() - t0;
    printf("  Save:    %.1f ms\n", save_ns / 1e6);

    /* Load */
    lcvdb_free(db);
    t0 = get_ns();
    db = lcvdb_load("/tmp/lcvdb_bench.db");
    double load_ns = get_ns() - t0;
    printf("  Load:    %.1f ms\n", load_ns / 1e6);

    /* Search after load */
    t0 = get_ns();
    for (int q = 0; q < 200; q++) {
        lcvdb_search(db, queries + (size_t)q * dim, 10, ids, scores);
    }
    double search2_ns = get_ns() - t0;
    printf("  Search (post-load): %.3f ms/query\n", search2_ns / 200 / 1e6);

    /* Memory estimate */
    size_t mem = (size_t)lcvdb_count(db) * (128 + 1792 + 192 + 8) + sizeof(void*) * 6;
    printf("  Memory:  ~%.1f MB (%u vectors)\n", mem / 1e6, lcvdb_count(db));

    lcvdb_free(db);
    unlink("/tmp/lcvdb_bench.db");
    free(vectors);
    free(queries);
    printf("\n");
}

int main(void) {
    printf("=== LCVDB Library Benchmark ===\n\n");

    bench_scale(1000,  768);
    bench_scale(10000, 768);
    bench_scale(65000, 768);
    bench_scale(1000,  1024);
    bench_scale(10000, 1024);
    bench_scale(65000, 1024);

    return 0;
}
