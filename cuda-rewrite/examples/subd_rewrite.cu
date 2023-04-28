#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <cuda.h>

#define LOG(fmt, ...) fprintf(stdout, fmt "\n", ##__VA_ARGS__); fflush(stdout);

#include "CatmullClark.h"

/*******************************************************************************
 * ExportToObj -- Exports subd to the OBJ file format
 *
 */
void
ExportToObj(
    const cc_Subd *subd,
    int32_t depth,
    const char *filename
) {
    const cc_Mesh *cage = subd->cage;
    const int32_t vertexPointCount = ccm_VertexCountAtDepth(cage, depth);
    const int32_t faceCount = ccm_FaceCountAtDepth(cage, depth);
    FILE *pf = fopen(filename, "w");

    // write vertices
    fprintf(pf, "# Vertices\n");
    if (depth == 0) {
        const int32_t vertexUvCount = ccm_UvCount(cage);

        for (int32_t vertexID = 0; vertexID < vertexPointCount; ++vertexID) {
            const float *v = ccm_VertexPoint(cage, vertexID).array;

            fprintf(pf, "v %f %f %f\n", v[0], v[1], v[2]);
        }

        for (int32_t vertexID = 0; vertexID < vertexUvCount; ++vertexID) {
            const float *v = ccm_Uv(cage, vertexID).array;

            fprintf(pf, "vt %f %f\n", v[0], v[1]);
        }
    } else {
        const int32_t halfedgeCount = ccm_HalfedgeCountAtDepth(cage, depth);

        for (int32_t vertexID = 0; vertexID < vertexPointCount; ++vertexID) {
            const float *v = ccs_VertexPoint(subd, vertexID, depth).array;

            fprintf(pf, "v %f %f %f\n", v[0], v[1], v[2]);
        }

#ifndef CC_DISABLE_UV
        for (int32_t halfedgeID = 0; halfedgeID < halfedgeCount; ++halfedgeID) {
            const float *uv = ccs_HalfedgeVertexUv(subd, halfedgeID, depth).array;

            fprintf(pf, "vt %f %f\n", uv[0], uv[1]);
        }
#endif
    }
    fprintf(pf, "\n");

    // write topology
    fprintf(pf, "# Topology\n");
    if (depth == 0) {
        for (int32_t faceID = 0; faceID < faceCount; ++faceID) {
            const int32_t halfEdgeID = ccm_FaceToHalfedgeID(cage, faceID);

            fprintf(pf,
                    "f %i/%i",
                    ccm_HalfedgeVertexID(cage, halfEdgeID) + 1,
                    ccm_HalfedgeUvID(cage, halfEdgeID) + 1);

            for (int32_t halfEdgeIt = ccm_HalfedgeNextID(cage, halfEdgeID);
                         halfEdgeIt != halfEdgeID;
                         halfEdgeIt = ccm_HalfedgeNextID(cage, halfEdgeIt)) {
                fprintf(pf,
                        " %i/%i",
                        ccm_HalfedgeVertexID(cage, halfEdgeIt) + 1,
                        ccm_HalfedgeUvID(cage, halfEdgeIt) + 1);
            }
            fprintf(pf, "\n");
        }
    } else {
        for (int32_t faceID = 0; faceID < faceCount; ++faceID) {
#ifndef CC_DISABLE_UV
            fprintf(pf,
                    "f %i/%i %i/%i %i/%i %i/%i\n",
                    ccs_HalfedgeVertexID(subd, 4 * faceID + 0, depth) + 1,
                    4 * faceID + 1,
                    ccs_HalfedgeVertexID(subd, 4 * faceID + 1, depth) + 1,
                    4 * faceID + 2,
                    ccs_HalfedgeVertexID(subd, 4 * faceID + 2, depth) + 1,
                    4 * faceID + 3,
                    ccs_HalfedgeVertexID(subd, 4 * faceID + 3, depth) + 1,
                    4 * faceID + 4);
#else
            fprintf(pf,
                    "f %i %i %i %i\n",
                    ccs_HalfedgeVertexID(subd, 4 * faceID + 0, depth) + 1,
                    ccs_HalfedgeVertexID(subd, 4 * faceID + 1, depth) + 1,
                    ccs_HalfedgeVertexID(subd, 4 * faceID + 2, depth) + 1,
                    ccs_HalfedgeVertexID(subd, 4 * faceID + 3, depth) + 1);
#endif
        }
        fprintf(pf, "\n");
    }

    fclose(pf);
}

typedef struct {
    double min, max, median, mean;
} BenchStats;

static int CompareCallback(const void * a, const void * b)
{
    if (*(double*)a > *(double*)b) {
        return 1;
    } else if (*(double*)a < *(double*)b) {
        return -1;
    } else {
        return 0;
    }
}

BenchStats Bench(void (*SubdCallback)(cc_Subd *subd), cc_Subd *subd)
{
#ifdef FLAG_BENCH
    const int32_t runCount = 100;
#else
    const int32_t runCount = 1;
#endif
#ifdef _WIN32
    DWORD startTime, stopTime;
#else
    struct timespec startTime, stopTime;
#endif
    double *times = (double *)malloc(sizeof(*times) * runCount);
    double timesTotal = 0.0;
    BenchStats stats = {0.0, 0.0, 0.0, 0.0};

    for (int32_t runID = 0; runID < runCount; ++runID) {
        double time = 0.0;

#ifdef _WIN32
        startTime = GetTickCount();
#else
        clock_gettime(CLOCK_MONOTONIC, &startTime);
#endif
        (*SubdCallback)(subd);
        cudaDeviceSynchronize();
#ifdef _WIN32
        stopTime = GetTickCount();
        time = (stopTime - startTime) / 1e3;
#else
        clock_gettime(CLOCK_MONOTONIC, &stopTime);

        time = (stopTime.tv_sec - startTime.tv_sec);
        time+= (stopTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;
#endif
        times[runID] = time;
        timesTotal+= time;
    }

    qsort(times, runCount, sizeof(times[0]), &CompareCallback);

    stats.min = times[0];
    stats.max = times[runCount - 1];
    stats.median = times[runCount / 2];
    stats.mean = timesTotal / runCount;

    free(times);

    return stats;
}

int main(int argc, char **argv)
{
    const char *filename = "./Kitchen_PUP.ccm";
    int32_t maxDepth = 4;
#ifdef FLAG_BENCH
    int32_t exportToObj = 0;
#else
    int32_t exportToObj = 1;
#endif
    cc_Mesh *cage = NULL;
    cc_Subd *subd = NULL;

    if (argc > 1) {
        filename = argv[1];
    }

    if (argc > 2) {
        maxDepth = atoi(argv[2]);
    }

    if (argc > 3) {
        exportToObj = atoi(argv[3]);
    }

    cage = ccm_Load(filename);

    if (!cage) {
        return -1;
    }

    subd = ccs_Create(cage, maxDepth);

    if (!subd) {
        ccm_Release(cage);

        return -1;
    }

    LOG("Refining... I have changed the code");
    // {
    //     const BenchStats stats = Bench(&ccs_RefineCreases, subd);

    //     LOG("Creases      -- median/mean/min/max (ms): %f / %f / %f / %f",
    //         stats.median * 1e3,
    //         stats.mean * 1e3,
    //         stats.min * 1e3,
    //         stats.max * 1e3);
    // }

    {
        const BenchStats stats = Bench(&ccs_RefineHalfedges, subd);

        LOG("Halfedges    -- median/mean/min/max (ms): %f / %f / %f / %f",
            stats.median * 1e3,
            stats.mean * 1e3,
            stats.min * 1e3,
            stats.max * 1e3);
    }

    // {
    //     const BenchStats stats = Bench(&ccs_RefineVertexPoints_Scatter, subd);

    //     LOG("VertexPoints -- median/mean/min/max (ms): %f / %f / %f / %f",
    //         stats.median * 1e3,
    //         stats.mean * 1e3,
    //         stats.min * 1e3,
    //         stats.max * 1e3);
    // }

// #ifndef CC_DISABLE_UV
//     {
//         const BenchStats stats = Bench(&ccs_RefineVertexUvs, subd);

//         LOG("VertexUvs    -- median/mean/min/max (ms): %f / %f / %f / %f",
//             stats.median * 1e3,
//             stats.mean * 1e3,
//             stats.min * 1e3,
//             stats.max * 1e3);
//     }
// #endif
    
    if (exportToObj > 0) {
        char buffer[64];

        LOG("Exporting...");
        for (int32_t depth = 0; depth <= maxDepth; ++depth) {
            sprintf(buffer, "subd_%01i_halfedges.obj", depth);

            ExportToObj(subd, depth, buffer);
            LOG("Level %i: done.", depth);
        }
    }

    LOG("All done!");

    // ccm_Release(cage);
    // ccs_Release(subd);

    return 0;
}
