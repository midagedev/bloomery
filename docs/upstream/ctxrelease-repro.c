// Minimal reproducer: does releasing two primary contexts crash the driver once host-mapped,
// portable pages were used across them?
//
// The engine's case (bloomery stagewin teardown): stage context on the A6000 (retained first), expert tier
// context on the 3090. The tier's page is cuMemHostAlloc(PORTABLE|DEVICEMAP) in the 3090's context and is
// addressed by the A6000's launches; the tier's rows are a page of the A6000's context written by the 3090.
// Each side's stream waits on and writes words of the other side's page with stream memory operations,
// eagerly and in captured graphs. Everything is freed and every stream is idle; then the 3090's primary
// context is released, then the A6000's, and the second release faults inside the driver.
//
// usage: ctxrepro [cross=0|1] [memop=0|1] [graph=0|1] [order=BA|AB] [free=0|1]
//   cross  1: each side touches the other side's page (default); 0: only its own
//   memop  1: stream wait/write value ops (default); 0: no device access to the pages at all
//   graph  1: the ops also run once as a captured, instantiated graph (default); 0: eager only
//   order  BA: release the 3090 (B) first, then the A6000 (A) (default, the engine's); AB: the reverse
//   free   1: cuMemFreeHost both pages before the releases (default); 0: leave them to the release
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = 0; cuGetErrorString(r_, &s_); \
    fprintf(stderr, "FAIL %s: %d %s (line %d)\n", #x, (int)r_, s_ ? s_ : "?", __LINE__); exit(2); } } while (0)

static int arg(int argc, char **argv, const char *k, int dflt) {
    size_t n = strlen(k);
    for (int i = 1; i < argc; i++)
        if (!strncmp(argv[i], k, n) && argv[i][n] == '=') return atoi(argv[i] + n + 1);
    return dflt;
}
static const char *sarg(int argc, char **argv, const char *k, const char *dflt) {
    size_t n = strlen(k);
    for (int i = 1; i < argc; i++)
        if (!strncmp(argv[i], k, n) && argv[i][n] == '=') return argv[i] + n + 1;
    return dflt;
}

static CUdevice by_name(const char *want) {
    int n = 0;
    CK(cuDeviceGetCount(&n));
    for (int i = 0; i < n; i++) {
        CUdevice d;
        char name[256];
        CK(cuDeviceGet(&d, i));
        CK(cuDeviceGetName(name, sizeof name, d));
        if (strstr(name, want)) return d;
    }
    fprintf(stderr, "FAIL: no device named like %s\n", want);
    exit(2);
}

// On `s`: write `v` to `dst`, then wait until `wait` >= 1 (the host raises it before syncing).
static void ops(CUstream s, CUdeviceptr dst, CUdeviceptr wait, unsigned v, int memop) {
    if (!memop) return;
    CK(cuStreamWriteValue32(s, dst, v, 0));
    CK(cuStreamWaitValue32(s, wait, 1, CU_STREAM_WAIT_VALUE_GEQ));
}

int main(int argc, char **argv) {
    int cross = arg(argc, argv, "cross", 1), memop = arg(argc, argv, "memop", 1);
    int graph = arg(argc, argv, "graph", 1), dofree = arg(argc, argv, "free", 1);
    const char *order = sarg(argc, argv, "order", "BA");
    printf("ctxrepro cross=%d memop=%d graph=%d order=%s free=%d\n", cross, memop, graph, order, dofree);
    fflush(stdout);

    CK(cuInit(0));
    CUdevice da = by_name("A6000"), db = by_name("3090");
    CUcontext ca, cb;
    CK(cuDevicePrimaryCtxRetain(&ca, da)); // the stage, retained first
    CK(cuDevicePrimaryCtxRetain(&cb, db)); // the tier

    const size_t bytes = 1 << 16;
    void *pa, *pb; // pa: the A6000's page (the tier rows); pb: the 3090's page (the tier page)
    CK(cuCtxSetCurrent(ca));
    CK(cuMemHostAlloc(&pa, bytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
    CK(cuCtxSetCurrent(cb));
    CK(cuMemHostAlloc(&pb, bytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
    memset(pa, 0, bytes);
    memset(pb, 0, bytes);

    // Device addresses of both pages in both contexts.
    CUdeviceptr pa_a, pb_a, pa_b, pb_b;
    CK(cuCtxSetCurrent(ca));
    CK(cuMemHostGetDevicePointer(&pa_a, pa, 0));
    CK(cuMemHostGetDevicePointer(&pb_a, pb, 0));
    CK(cuCtxSetCurrent(cb));
    CK(cuMemHostGetDevicePointer(&pa_b, pa, 0));
    CK(cuMemHostGetDevicePointer(&pb_b, pb, 0));

    CUstream sa, sb;
    CK(cuCtxSetCurrent(ca));
    CK(cuStreamCreate(&sa, CU_STREAM_NON_BLOCKING));
    CK(cuCtxSetCurrent(cb));
    CK(cuStreamCreate(&sb, CU_STREAM_NON_BLOCKING));

    // Word 0 of each page: what the other side writes; word 16: the host's go for each side.
    volatile unsigned *wa = (volatile unsigned *)pa, *wb = (volatile unsigned *)pb;
    for (int round = 0; round < 3; round++) {
        wa[16] = 0;
        wb[16] = 0;
        CK(cuCtxSetCurrent(ca));
        ops(sa, cross ? pb_a : pa_a, pa_a + 64, 100 + round, memop);
        CK(cuCtxSetCurrent(cb));
        ops(sb, cross ? pa_b : pb_b, pb_b + 64, 200 + round, memop);
        wa[16] = 1;
        wb[16] = 1;
        CK(cuCtxSetCurrent(ca));
        CK(cuStreamSynchronize(sa));
        CK(cuCtxSetCurrent(cb));
        CK(cuStreamSynchronize(sb));
    }
    if (memop) printf("eager: pa[0]=%u pb[0]=%u\n", wa[0], wb[0]);

    if (graph && memop) {
        CUgraph g[2];
        CUgraphExec e[2];
        CUstream ss[2] = {sa, sb};
        CUcontext cc[2] = {ca, cb};
        CUdeviceptr dst[2] = {cross ? pb_a : pa_a, cross ? pa_b : pb_b};
        CUdeviceptr wt[2] = {pa_a + 64, pb_b + 64};
        wa[16] = 1;
        wb[16] = 1;
        for (int i = 0; i < 2; i++) {
            CK(cuCtxSetCurrent(cc[i]));
            CK(cuStreamBeginCapture(ss[i], CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
            ops(ss[i], dst[i], wt[i], 300 + i, 1);
            CK(cuStreamEndCapture(ss[i], &g[i]));
            CK(cuGraphInstantiateWithFlags(&e[i], g[i], 0));
            CK(cuGraphLaunch(e[i], ss[i]));
            CK(cuStreamSynchronize(ss[i]));
        }
        printf("graph: pa[0]=%u pb[0]=%u\n", wa[0], wb[0]);
        for (int i = 0; i < 2; i++) {
            CK(cuCtxSetCurrent(cc[i]));
            CK(cuGraphExecDestroy(e[i]));
            CK(cuGraphDestroy(g[i]));
        }
    }

    // Teardown as the engine's: every stream idle, the pages freed, the streams destroyed, then the releases.
    CK(cuCtxSetCurrent(cb));
    CK(cuStreamSynchronize(sb));
    if (dofree) CK(cuMemFreeHost(pb));
    CK(cuStreamDestroy(sb));
    CK(cuCtxSetCurrent(ca));
    CK(cuStreamSynchronize(sa));
    if (dofree) CK(cuMemFreeHost(pa));
    CK(cuStreamDestroy(sa));
    printf("freed; releasing %s\n", order);
    fflush(stdout);
    if (!strcmp(order, "BA")) {
        CK(cuDevicePrimaryCtxRelease(db));
        printf("released B (3090)\n");
        fflush(stdout);
        CK(cuDevicePrimaryCtxRelease(da));
        printf("released A (A6000)\n");
    } else {
        CK(cuDevicePrimaryCtxRelease(da));
        printf("released A (A6000)\n");
        fflush(stdout);
        CK(cuDevicePrimaryCtxRelease(db));
        printf("released B (3090)\n");
    }
    printf("ok\n");
    return 0;
}
