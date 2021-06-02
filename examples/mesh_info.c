
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define CBF_IMPLEMENTATION
#include "ConcurrentBitField.h"
#undef CBF_IMPLEMENTATION

#define CC_IMPLEMENTATION
#include "CatmullClark.h"

#define LOG(fmt, ...) fprintf(stdout, fmt "\n", ##__VA_ARGS__); fflush(stdout);

static void usage(const char *appname)
{
    LOG("usage: %s path_to_ccm maxDepth", appname);
}

double ByteToGiByte(int64_t size)
{
    return (double)size / (1 << 30);
}

int main(int argc, char **argv)
{
    int32_t maxDepth;
    cc_Mesh *mesh;
    int32_t nonQuadCount = 0;
    int32_t creaseCount = 0;
    int32_t boundaryCount = 0;

    if (argc < 3) {
        usage(argv[0]);
        return 0;
    }

    mesh = ccm_Load(argv[1]);
    maxDepth = atoi(argv[2]);

    // boundaries
    boundaryCount = 2 * ccm_EdgeCount(mesh) - ccm_HalfedgeCount(mesh);

    // count non quads
    for (int32_t faceID = 0; faceID < ccm_FaceCount(mesh); ++faceID) {
        const int32_t halfedgeID = ccm_FaceToHalfedgeID(mesh, faceID);
        int32_t halfedgeIt = ccm_HalfedgeNextID(mesh, halfedgeID);
        int32_t cycleLength = 1;

        while (halfedgeIt != halfedgeID) {
            ++cycleLength;
            halfedgeIt = ccm_HalfedgeNextID(mesh, halfedgeIt);
        }

        if (cycleLength != 4)
            ++nonQuadCount;
    }

    // creases
    for (int32_t edgeID = 0; edgeID < ccm_EdgeCount(mesh); ++edgeID) {
        if (ccm_CreaseSharpness(mesh, edgeID)) {
            ++creaseCount;
        }
    }
    creaseCount-= boundaryCount;

    LOG("(non Quads: %i; boundaries: %i; creases: %i)",
        nonQuadCount,
        boundaryCount,
        creaseCount);
    LOG("(UVs: %i)", ccm_UvCount(mesh));

    for (int32_t depth = 0; depth <= maxDepth; ++depth) {
        LOG("depth %i: H= %i F= %i E= %i V= %i C= %i",
            depth,
            ccm_HalfedgeCountAtDepth(mesh, depth),
            ccm_FaceCountAtDepth(mesh, depth),
            ccm_EdgeCountAtDepth(mesh, depth),
            ccm_VertexCountAtDepth(mesh, depth),
            ccm_CreaseCountAtDepth(mesh, depth));
    }

    for (int32_t depth = 0; depth <= maxDepth; ++depth) {
        int32_t Href = 0, Vref = 0, Fref = 0, Eref = 0, Cref = 0;

        for (int32_t d = 1; d <= depth; ++d) {
            Href+= ccm_HalfedgeCountAtDepth(mesh, d);
            Vref+= ccm_VertexCountAtDepth(mesh, d);
            Eref+= ccm_EdgeCountAtDepth(mesh, d);
            Fref+= ccm_FaceCountAtDepth(mesh, d);
            Cref+= ccm_CreaseCountAtDepth(mesh, d);
        }

        LOG("depth %i: Hcum= %i (ref: %i)\n"
            "         Fcum= %i (ref: %i)\n"
            "         Ecum= %i (ref: %i)\n"
            "         Vcum= %i (ref: %i)\n"
            "         Ccum= %i (ref: %i)\n",
            depth,
            ccs_CumulativeHalfedgeCountAtDepth(mesh, depth),
            Href,
            ccs_CumulativeFaceCountAtDepth(mesh, depth),
            Fref,
            ccs_CumulativeEdgeCountAtDepth(mesh, depth),
            Eref,
            ccs_CumulativeVertexCountAtDepth(mesh, depth),
            Vref,
            ccs_CumulativeCreaseCountAtDepth(mesh, depth),
            Cref);
    }

    ccm_Release(mesh);

    return 1;
}
