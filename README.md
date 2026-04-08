# LCVDB — Embeddable Vector Search Library

**42 KB. 7 functions. 100% recall. Zero configuration.**

LCVDB is an embeddable C library for nearest-neighbor vector search. It exploits a discovered property of high-dimensional trit-quantized space: at ≥1024 dimensions, the Hamming-nearest trit fingerprint IS the true nearest neighbor, always. No index structure needed.

## API

```c
#include "lcvdb.h"

lcvdb_t *db = lcvdb_create(768, 100000);     // dim, initial capacity
lcvdb_insert(db, id, float_vector);           // O(1) — pack + append
lcvdb_search(db, query, 10, ids, scores);     // O(N) scan, 100% R@1
lcvdb_save(db, "vectors.db");
lcvdb_free(db);

// Later:
lcvdb_t *db = lcvdb_load("vectors.db");
```

## Build

```bash
make native     # AVX2 (42 KB) — fastest
make scalar     # Portable (40 KB) — any CPU
make wasm       # WASM + SIMD128 — browser
make test       # Run basic test
```

## Targets

| Target | Size | SIMD | Use case |
|--------|------|------|----------|
| `liblcvdb.a` | 42 KB | AVX2 | Native x86 server/desktop |
| `liblcvdb_scalar.a` | 40 KB | None | ARM, RISC-V, any CPU |
| `lcvdb.wasm` | TBD | WASM SIMD128 | Browser, Edge, Cloudflare Workers |

## The Science

At 1024 dimensions, trit Hamming distance is a perfect nearest-neighbor proxy:

| N | Dim | max_rank | R@1 |
|---|-----|----------|-----|
| 65K | 1024 | **0** | 100% |
| 41.5M | 1024 | **0** | 100% |

Zero rank violations across 800 queries at four corpus sizes. The concentration of measure in high-dimensional trit space eliminates the need for any index structure.

See the [main LCVDB repo](https://github.com/anjaustin/lcvdb) for the full paper, server, and research tooling.

## License

MIT
