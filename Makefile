# LCVDB Library — Portable build (AVX2 / WASM SIMD / scalar)

SRCDIR = src
BUILDDIR = build

# Source files
SRC = $(SRCDIR)/lcvdb.c $(SRCDIR)/init_ternary.c $(SRCDIR)/build_ternary.c \
      $(SRCDIR)/distance_avx2.c $(SRCDIR)/sce.c $(SRCDIR)/persist_ternary.c

HEADERS = $(SRCDIR)/platform.h $(SRCDIR)/lcvdb_ternary.h $(SRCDIR)/sce.h \
          include/lcvdb.h

# === Native x86 (AVX2) ===
CC = gcc
CFLAGS_AVX2 = -O2 -Wall -mavx2 -mpopcnt -mfma -mbmi2 -I$(SRCDIR) -Iinclude
CFLAGS_SCALAR = -O2 -Wall -I$(SRCDIR) -Iinclude

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Static library (AVX2 — native x86)
$(BUILDDIR)/liblcvdb.a: $(SRC) $(HEADERS) | $(BUILDDIR)
	$(CC) $(CFLAGS_AVX2) -c $(SRCDIR)/lcvdb.c -o $(BUILDDIR)/lcvdb.o
	$(CC) $(CFLAGS_AVX2) -c $(SRCDIR)/init_ternary.c -o $(BUILDDIR)/init.o
	$(CC) $(CFLAGS_AVX2) -c $(SRCDIR)/build_ternary.c -o $(BUILDDIR)/build.o
	$(CC) $(CFLAGS_AVX2) -c $(SRCDIR)/distance_avx2.c -o $(BUILDDIR)/dist.o
	$(CC) $(CFLAGS_AVX2) -c $(SRCDIR)/sce.c -o $(BUILDDIR)/sce.o
	$(CC) $(CFLAGS_AVX2) -c $(SRCDIR)/persist_ternary.c -o $(BUILDDIR)/persist.o
	ar rcs $@ $(BUILDDIR)/lcvdb.o $(BUILDDIR)/init.o $(BUILDDIR)/build.o \
	          $(BUILDDIR)/dist.o $(BUILDDIR)/sce.o $(BUILDDIR)/persist.o
	@echo "Built $@ ($$(wc -c < $@) bytes)"

# Static library (scalar — portable)
$(BUILDDIR)/liblcvdb_scalar.a: $(SRC) $(HEADERS) | $(BUILDDIR)
	$(CC) $(CFLAGS_SCALAR) -c $(SRCDIR)/lcvdb.c -o $(BUILDDIR)/lcvdb_s.o
	$(CC) $(CFLAGS_SCALAR) -c $(SRCDIR)/init_ternary.c -o $(BUILDDIR)/init_s.o
	$(CC) $(CFLAGS_SCALAR) -c $(SRCDIR)/build_ternary.c -o $(BUILDDIR)/build_s.o
	$(CC) $(CFLAGS_SCALAR) -c $(SRCDIR)/distance_avx2.c -o $(BUILDDIR)/dist_s.o
	$(CC) $(CFLAGS_SCALAR) -c $(SRCDIR)/sce.c -o $(BUILDDIR)/sce_s.o
	$(CC) $(CFLAGS_SCALAR) -c $(SRCDIR)/persist_ternary.c -o $(BUILDDIR)/persist_s.o
	ar rcs $@ $(BUILDDIR)/lcvdb_s.o $(BUILDDIR)/init_s.o $(BUILDDIR)/build_s.o \
	          $(BUILDDIR)/dist_s.o $(BUILDDIR)/sce_s.o $(BUILDDIR)/persist_s.o
	@echo "Built $@ ($$(wc -c < $@) bytes)"

# === WASM (Emscripten) ===
EMCC = emcc
CFLAGS_WASM = -O3 -msimd128 -I$(SRCDIR) -Iinclude -D__EMSCRIPTEN__

$(BUILDDIR)/lcvdb.wasm: $(SRC) $(HEADERS) | $(BUILDDIR)
	$(EMCC) $(CFLAGS_WASM) \
		-s WASM=1 \
		-s EXPORTED_FUNCTIONS='["_lcvdb_create","_lcvdb_free","_lcvdb_insert","_lcvdb_search","_lcvdb_count","_lcvdb_delete","_malloc","_free"]' \
		-s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue"]' \
		-s ALLOW_MEMORY_GROWTH=1 \
		-s INITIAL_MEMORY=16777216 \
		-o $@ $(SRC) -lm
	@echo "Built $@ ($$(wc -c < $@) bytes)"

# Convenience targets
.PHONY: native scalar wasm clean test

native: $(BUILDDIR)/liblcvdb.a
scalar: $(BUILDDIR)/liblcvdb_scalar.a
wasm: $(BUILDDIR)/lcvdb.wasm

# Test (native)
$(BUILDDIR)/test_basic: test/test_basic.c $(BUILDDIR)/liblcvdb.a
	$(CC) $(CFLAGS_AVX2) -Iinclude -o $@ $< -L$(BUILDDIR) -llcvdb -lm

test: $(BUILDDIR)/test_basic
	./$(BUILDDIR)/test_basic

clean:
	rm -rf $(BUILDDIR)
