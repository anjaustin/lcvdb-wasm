/* test_basic.c — Verify the public API works end-to-end. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lcvdb.h"

int main(void) {
    printf("=== LCVDB Library Basic Test ===\n\n");

    /* Create */
    lcvdb_t *db = lcvdb_create(128, 1000);
    if (!db) { printf("FAIL: create\n"); return 1; }
    printf("Created: dim=128, capacity=1000\n");

    /* Insert 100 random vectors */
    uint32_t seed = 42;
    for (int i = 0; i < 100; i++) {
        float vec[128];
        for (int d = 0; d < 128; d++) {
            seed = seed * 1103515245 + 12345;
            vec[d] = ((float)(seed >> 16) / 32768.0f) - 1.0f;
        }
        uint32_t nid = lcvdb_insert(db, (uint32_t)i, vec);
        if (nid == UINT32_MAX) { printf("FAIL: insert %d\n", i); return 1; }
    }
    printf("Inserted: %u vectors\n", lcvdb_count(db));

    /* Search */
    float query[128];
    seed = 42; /* Same seed as first vector → should find itself */
    for (int d = 0; d < 128; d++) {
        seed = seed * 1103515245 + 12345;
        query[d] = ((float)(seed >> 16) / 32768.0f) - 1.0f;
    }

    uint32_t ids[5];
    float scores[5];
    int found = lcvdb_search(db, query, 5, ids, scores);
    printf("Search: found=%d, top_id=%u, top_score=%.4f\n", found, ids[0], scores[0]);

    if (found < 1) { printf("FAIL: no results\n"); return 1; }
    if (scores[0] < 0.99f) { printf("FAIL: self-search score too low (%.4f)\n", scores[0]); return 1; }

    /* Delete */
    lcvdb_delete(db, 0);
    printf("Deleted vector 0, count=%u\n", lcvdb_count(db));

    /* Save and reload */
    if (lcvdb_save(db, "/tmp/lcvdb_test.db") != 0) { printf("FAIL: save\n"); return 1; }
    printf("Saved to /tmp/lcvdb_test.db\n");

    lcvdb_free(db);

    db = lcvdb_load("/tmp/lcvdb_test.db");
    if (!db) { printf("FAIL: load\n"); return 1; }
    printf("Loaded: count=%u\n", lcvdb_count(db));

    /* Search after reload */
    found = lcvdb_search(db, query, 5, ids, scores);
    printf("Search after reload: found=%d, top_score=%.4f\n", found, scores[0]);

    lcvdb_free(db);
    unlink("/tmp/lcvdb_test.db");

    /* Test growable capacity */
    printf("\n--- Grow test ---\n");
    db = lcvdb_create(128, 10); /* tiny initial capacity */
    if (!db) { printf("FAIL: create small\n"); return 1; }
    for (int i = 0; i < 100; i++) {
        float vec[128];
        uint32_t s = (uint32_t)(i + 1000);
        for (int d = 0; d < 128; d++) { s = s * 1103515245 + 12345; vec[d] = ((float)(s >> 16) / 32768.0f) - 1.0f; }
        uint32_t nid = lcvdb_insert(db, (uint32_t)(i + 1000), vec);
        if (nid == UINT32_MAX) { printf("FAIL: insert %d during grow\n", i); return 1; }
    }
    printf("Inserted 100 into capacity-10 db: count=%u (grew to fit)\n", lcvdb_count(db));
    if (lcvdb_count(db) != 100) { printf("FAIL: expected 100, got %u\n", lcvdb_count(db)); return 1; }

    /* Test external ID delete */
    lcvdb_delete(db, 1050); /* delete by external ID */
    /* Search for the deleted vector — should not be top result */
    {
        float vec[128];
        uint32_t s = 1050;
        for (int d = 0; d < 128; d++) { s = s * 1103515245 + 12345; vec[d] = ((float)(s >> 16) / 32768.0f) - 1.0f; }
        found = lcvdb_search(db, vec, 1, ids, scores);
        printf("After delete 1050: search found=%d, top_id=%u (should NOT be 1050's node)\n", found, ids[0]);
    }

    lcvdb_free(db);

    printf("\nPASS\n");
    return 0;
}
