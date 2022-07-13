#define CC_IMPLEMENTATION
#include "CatmullClark.h"

#define CBF_IMPLEMENTATION
#include "ConcurrentBitField.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifndef CC_ASSERT
#    include <assert.h>
#    define CC_ASSERT(x) assert(x)
#endif

#ifndef CC_LOG
#    include <stdio.h>
#    define CC_LOG(format, ...) do { fprintf(stdout, format "\n", ##__VA_ARGS__); fflush(stdout); } while(0)
#endif

#ifndef CC_MALLOC
#    include <stdlib.h>
#    define CC_MALLOC(x) (malloc(x))
#    define CC_FREE(x) (free(x))
#else
#    ifndef CC_FREE
#        error CC_MALLOC defined without CC_FREE
#    endif
#endif

#ifndef CC_MEMCPY
#    include <string.h>
#    define CC_MEMCPY(dest, src, count) memcpy(dest, src, count)
#endif

#ifndef CC_MEMSET
#    include <string.h>
#    define CC_MEMSET(ptr, value, num) memset(ptr, value, num)
#endif

#ifndef _OPENMP
#   ifndef CC_ATOMIC
#       define CC_ATOMIC
#   endif
#   ifndef CC_PARALLEL_FOR
#       define CC_PARALLEL_FOR
#   endif
#   ifndef CC_BARRIER
#       define CC_BARRIER
#   endif
#else
#   if defined(_WIN32)
#       ifndef CC_ATOMIC
#           define CC_ATOMIC          __pragma("omp atomic" )
#       endif
#       ifndef CC_PARALLEL_FOR
#           define CC_PARALLEL_FOR    __pragma("omp parallel for")
#       endif
#       ifndef CC_BARRIER
#           define CC_BARRIER         __pragma("omp barrier")
#       endif
#   else
#       ifndef CC_ATOMIC
#           define CC_ATOMIC          _Pragma("omp atomic" )
#       endif
#       ifndef CC_PARALLEL_FOR
#           define CC_PARALLEL_FOR    _Pragma("omp parallel for")
#       endif
#       ifndef CC_BARRIER
#           define CC_BARRIER         _Pragma("omp barrier")
#       endif
#   endif
#endif


/*******************************************************************************
 * ComputeTwins -- Computes the twin of each half edge
 *
 * This routine is what effectively converts a traditional "indexed mesh"
 * into a halfedge mesh (in the case where all the primitives are the same).
 *
 */
typedef struct {
    int32_t halfedgeID;
    uint64_t hashID;
} TwinComputationData;

static int32_t
BinarySearch(
    const TwinComputationData *data,
    int64_t hashID,
    int32_t beginID,
    int32_t endID
) {
    int32_t midID;

    if (beginID > endID)
       return -1; // not found

    midID = (beginID + endID) / 2;

    if (data[midID].hashID == hashID) {
        return data[midID].halfedgeID;
    } else if (hashID > data[midID].hashID) {
        return BinarySearch(data, hashID, midID + 1, endID);
    } else {
        return BinarySearch(data, hashID, beginID, midID - 1);
    }
}

static void
SortTwinComputationData(TwinComputationData *array, uint32_t arraySize)
{
    for (uint32_t d2 = 1u; d2 < arraySize; d2*= 2u) {
        for (uint32_t d1 = d2; d1 >= 1u; d1/= 2u) {
            const uint32_t mask = (0xFFFFFFFEu * d1);

CC_PARALLEL_FOR
            for (uint32_t i = 0; i < (arraySize / 2); ++i) {
                const uint32_t i1 = ((i << 1) & mask) | (i & ~(mask >> 1));
                const uint32_t i2 = i1 | d1;
                const TwinComputationData t1 = array[i1];
                const TwinComputationData t2 = array[i2];
                const TwinComputationData min = t1.hashID < t2.hashID ? t1 : t2;
                const TwinComputationData max = t1.hashID < t2.hashID ? t2 : t1;

                if ((i & d2) == 0) {
                    array[i1] = min;
                    array[i2] = max;
                } else {
                    array[i1] = max;
                    array[i2] = min;
                }
            }
CC_BARRIER
        }
    }
}

static int32_t RoundUpToPowerOfTwo(int32_t x)
{
    x--;
    x|= x >>  1;
    x|= x >>  2;
    x|= x >>  4;
    x|= x >>  8;
    x|= x >> 16;
    x++;

    return x;
}

static void ComputeTwins(cc_Mesh *mesh)
{
    const int32_t halfedgeCount = ccm_HalfedgeCount(mesh);
    const int32_t vertexCount = ccm_VertexCount(mesh);
    const int32_t tableSize = RoundUpToPowerOfTwo(halfedgeCount);
    TwinComputationData *table = (TwinComputationData *)CC_MALLOC(tableSize * sizeof(*table));

    CC_PARALLEL_FOR
    for (int32_t halfedgeID = 0; halfedgeID < halfedgeCount; ++halfedgeID) {
        const int32_t nextID = ccm_HalfedgeNextID(mesh, halfedgeID);
        const int32_t v0 = ccm_HalfedgeVertexID(mesh, halfedgeID);
        const int32_t v1 = ccm_HalfedgeVertexID(mesh, nextID);

        table[halfedgeID].halfedgeID = halfedgeID;
        table[halfedgeID].hashID = (uint64_t)v0 + (uint64_t)vertexCount * v1;
    }
    CC_BARRIER

    CC_PARALLEL_FOR
    for (int32_t halfedgeID = halfedgeCount; halfedgeID < tableSize; ++halfedgeID) {
        table[halfedgeID].hashID = ~0ULL;
    }
    CC_BARRIER

    SortTwinComputationData(table, tableSize);

    CC_PARALLEL_FOR
    for (int32_t halfedgeID = 0; halfedgeID < halfedgeCount; ++halfedgeID) {
        const int32_t nextID = ccm_HalfedgeNextID(mesh, halfedgeID);
        const int32_t v0 = ccm_HalfedgeVertexID(mesh, halfedgeID);
        const int32_t v1 = ccm_HalfedgeVertexID(mesh, nextID);
        const int64_t hashID = (uint64_t)v1 + (uint64_t)vertexCount * v0;
        const int32_t twinID = BinarySearch(table, hashID, 0, halfedgeCount - 1);

        mesh->halfedges[halfedgeID].twinID = twinID;
    }
    CC_BARRIER

    CC_FREE(table);
}


/*******************************************************************************
 * ComputeCreaseNeighbors -- Computes the neighbors of each crease
 *
 */
static void ComputeCreaseNeighbors(cc_Mesh *mesh)
{
    const int32_t edgeCount = ccm_EdgeCount(mesh);

CC_PARALLEL_FOR
    for (int32_t edgeID = 0; edgeID < edgeCount; ++edgeID) {
        const float sharpness = ccm_CreaseSharpness(mesh, edgeID);

        if (sharpness > 0.0f) {
            const int32_t halfedgeID = ccm_EdgeToHalfedgeID(mesh, edgeID);
            const int32_t nextID = ccm_HalfedgeNextID(mesh, halfedgeID);
            int32_t prevCreaseCount = 0;
            int32_t prevCreaseID = -1;
            int32_t nextCreaseCount = 0;
            int32_t nextCreaseID = -1;
            int32_t halfedgeIt;

            for (halfedgeIt = ccm_NextVertexHalfedgeID(mesh, halfedgeID);
                 halfedgeIt != halfedgeID && halfedgeIt >= 0;
                 halfedgeIt = ccm_NextVertexHalfedgeID(mesh, halfedgeIt)) {
                const float s = ccm_HalfedgeSharpness(mesh, halfedgeIt);

                if (s > 0.0f) {
                    prevCreaseID = ccm_HalfedgeEdgeID(mesh, halfedgeIt);
                    ++prevCreaseCount;
                }
            }

            if (prevCreaseCount == 1 && halfedgeIt == halfedgeID) {
                mesh->creases[edgeID].prevID = prevCreaseID;
            }

            if (ccm_HalfedgeSharpness(mesh, nextID) > 0.0f) {
                nextCreaseID = ccm_HalfedgeEdgeID(mesh, nextID);
                ++nextCreaseCount;
            }

            for (halfedgeIt = ccm_NextVertexHalfedgeID(mesh, nextID);
                 halfedgeIt != nextID && halfedgeIt >= 0;
                 halfedgeIt = ccm_NextVertexHalfedgeID(mesh, halfedgeIt)) {
                const float s = ccm_HalfedgeSharpness(mesh, halfedgeIt);
                const int32_t twinID = ccm_HalfedgeTwinID(mesh, halfedgeIt);

                // twin check is to avoid counting for halfedgeID
                if (s > 0.0f && twinID != halfedgeID) {
                    nextCreaseID = ccm_HalfedgeEdgeID(mesh, halfedgeIt);
                    ++nextCreaseCount;
                }
            }

            if (nextCreaseCount == 1 && halfedgeIt == nextID) {
                mesh->creases[edgeID].nextID = nextCreaseID;
            }
        }
    }
CC_BARRIER
}


/*******************************************************************************
 * MakeBoundariesSharp -- Tags boundary edges as sharp
 *
 * Following the Pixar standard, we tag boundary halfedges as sharp.
 * See "Subdivision Surfaces in Character Animation" by DeRose et al.
 * Note that we tag the sharpness value to 16 as subdivision can't go deeper
 * without overflowing 32-bit integers.
 *
 */
static void MakeBoundariesSharp(cc_Mesh *mesh)
{
    const int32_t edgeCount = ccm_EdgeCount(mesh);

CC_PARALLEL_FOR
    for (int32_t edgeID = 0; edgeID < edgeCount; ++edgeID) {
        const int32_t halfedgeID = ccm_EdgeToHalfedgeID(mesh, edgeID);
        const int32_t twinID = ccm_HalfedgeTwinID(mesh, halfedgeID);

        if (twinID < 0) {
            mesh->creases[edgeID].sharpness = 16.0f;
        }
    }
CC_BARRIER
}


/*******************************************************************************
 * LoadFaceMappings -- Computes the mappings for the faces of the mesh
 *
 */
static int32_t FaceScroll(int32_t id, int32_t direction, int32_t maxValue)
{
    const int32_t n = maxValue - 1;
    const int32_t d = direction;
    const int32_t u = (d + 1) >> 1; // in [0, 1]
    const int32_t un = u * n; // precomputation

    return (id == un) ? (n - un) : (id + d);
}

static int32_t
ScrollFaceHalfedgeID(
    int32_t halfedgeID,
    int32_t halfedgeFaceBeginID,
    int32_t halfedgeFaceEndID,
    int32_t direction
) {
    const int32_t faceHalfedgeCount = halfedgeFaceEndID - halfedgeFaceBeginID;
    const int32_t localHalfedgeID = halfedgeID - halfedgeFaceBeginID;
    const int32_t nextHalfedgeID = FaceScroll(localHalfedgeID,
                                              direction,
                                              faceHalfedgeCount);

    return halfedgeFaceBeginID + nextHalfedgeID;
}

static void LoadFaceMappings(cc_Mesh *mesh, const cbf_BitField *faceIterator)
{
    const int32_t halfedgeCount = ccm_HalfedgeCount(mesh);
    const int32_t faceCount = cbf_BitCount(faceIterator) - 1;

    mesh->faceToHalfedgeIDs = (int32_t *)CC_MALLOC(sizeof(int32_t) * faceCount);
    mesh->faceCount = faceCount;

CC_PARALLEL_FOR
    for (int32_t halfedgeID = 0; halfedgeID  < halfedgeCount; ++halfedgeID) {
        const int32_t tmp = cbf_EncodeBit(faceIterator, halfedgeID);
        const int32_t faceID = tmp - (cbf_GetBit(faceIterator, halfedgeID) ^ 1);

        mesh->halfedges[halfedgeID].faceID = faceID;
    }
CC_BARRIER

CC_PARALLEL_FOR
    for (int32_t faceID = 0; faceID < faceCount; ++faceID) {
        mesh->faceToHalfedgeIDs[faceID] = cbf_DecodeBit(faceIterator, faceID);
    }
CC_BARRIER


CC_PARALLEL_FOR
    for (int32_t halfedgeID = 0; halfedgeID  < halfedgeCount; ++halfedgeID) {
        const int32_t faceID = mesh->halfedges[halfedgeID].faceID;
        const int32_t beginID = cbf_DecodeBit(faceIterator, faceID);
        const int32_t endID = cbf_DecodeBit(faceIterator, faceID + 1);
        const int32_t nextID = ScrollFaceHalfedgeID(halfedgeID, beginID, endID, +1);
        const int32_t prevID = ScrollFaceHalfedgeID(halfedgeID, beginID, endID, -1);

        mesh->halfedges[halfedgeID].nextID = nextID;
        mesh->halfedges[halfedgeID].prevID = prevID;
    }
CC_BARRIER
}


/*******************************************************************************
 * LoadEdgeMappings -- Computes the mappings for the edges of the mesh
 *
 * Catmull-Clark subdivision requires access to the edges of an input mesh.
 * Since we are dealing with a halfedge representation, we virtually
 * have to iterate the halfedges in a sparse way (an edge is a pair of
 * neighboring halfedges in the general case, except for boundary edges
 * where it only consists of a single halfedge).
 * This function builds a data-structure that allows to do just that:
 * for each halfedge pair, we only consider the one that has the largest
 * halfedgeID. This allows to treat boundary and regular edges seamlessly.
 *
 */
static void LoadEdgeMappings(cc_Mesh *mesh)
{
    const int32_t halfedgeCount = ccm_HalfedgeCount(mesh);
    cc_Halfedge *halfedges = mesh->halfedges;
    cbf_BitField *edgeIterator = cbf_Create(halfedgeCount);
    int32_t edgeCount;

CC_PARALLEL_FOR
    for (int32_t halfedgeID = 0; halfedgeID < halfedgeCount; ++halfedgeID) {
        int32_t twinID = ccm_HalfedgeTwinID(mesh, halfedgeID);
        int32_t bitValue = halfedgeID > twinID ? 1 : 0;

        cbf_SetBit(edgeIterator, halfedgeID, bitValue);
    }
CC_BARRIER

    cbf_Reduce(edgeIterator);
    edgeCount = cbf_BitCount(edgeIterator);

    mesh->edgeToHalfedgeIDs = (int32_t *)CC_MALLOC(sizeof(int32_t) * edgeCount);
    mesh->edgeCount = edgeCount;

CC_PARALLEL_FOR
    for (int32_t halfedgeID = 0; halfedgeID  < halfedgeCount; ++halfedgeID) {
        const int32_t twinID = ccm_HalfedgeTwinID(mesh, halfedgeID);
        const int32_t bitID = cc__Max(halfedgeID, twinID);

        halfedges[halfedgeID].edgeID = cbf_EncodeBit(edgeIterator, bitID);
    }
CC_BARRIER

CC_PARALLEL_FOR
    for (int32_t edgeID = 0; edgeID < edgeCount; ++edgeID) {
        mesh->edgeToHalfedgeIDs[edgeID] = cbf_DecodeBit(edgeIterator, edgeID);
    }
CC_BARRIER

    cbf_Release(edgeIterator);
}


/*******************************************************************************
 * LoadVertexHalfedges -- Computes an iterator over one halfedge per vertex
 *
 * Catmull-Clark subdivision requires access to the halfedges that surround
 * the vertices of an input mesh.
 * This function determines a halfedge ID that starts from a
 * given vertex within that vertex. We distinguish two cases:
 * 1- If the vertex is a lying on a boundary, we stored the halfedge that
 * allows for iteration in the forward sense.
 * 2- Otherwise we store the largest halfedge ID.
 *
 */
static void LoadVertexHalfedges(cc_Mesh *mesh)
{
    const int32_t halfedgeCount = ccm_HalfedgeCount(mesh);
    const int32_t vertexCount = ccm_VertexCount(mesh);

    mesh->vertexToHalfedgeIDs = (int32_t *)CC_MALLOC(sizeof(int32_t) * vertexCount);

CC_PARALLEL_FOR
    for (int32_t halfedgeID = 0; halfedgeID < halfedgeCount; ++halfedgeID) {
        const int32_t vertexID = ccm_HalfedgeVertexID(mesh, halfedgeID);
        int32_t maxHalfedgeID = halfedgeID;
        int32_t boundaryHalfedgeID = halfedgeID;
        int32_t iterator;

        for (iterator = ccm_NextVertexHalfedgeID(mesh, halfedgeID);
             iterator >= 0 && iterator != halfedgeID;
             iterator = ccm_NextVertexHalfedgeID(mesh, iterator)) {
            maxHalfedgeID = cc__Max(maxHalfedgeID, iterator);
            boundaryHalfedgeID = iterator;
        }

        // affect max halfedge ID to vertex
        if /*boundary involved*/ (iterator < 0) {
            if (halfedgeID == boundaryHalfedgeID) {
                mesh->vertexToHalfedgeIDs[vertexID] = boundaryHalfedgeID;
            }
        } else {
            if (halfedgeID == maxHalfedgeID) {
                mesh->vertexToHalfedgeIDs[vertexID] = maxHalfedgeID;
            }
        }
    }
CC_BARRIER
}


/*******************************************************************************
 * ObjReadFace -- Reads an OBJ face
 *
 * OBJ files can describe a face according to 4 different formats.
 * This function supports each format. It returns the number of vertices
 * read for the current face.
 *
 */
static int32_t ObjReadFace(const char *str, cc_Halfedge **halfedges)
{
    const char *formats[] = {"%d/%*d/%*d%n", "%d//%*d%n", "%d/%*d%n", "%d%n"};
    int32_t halfedgeCount = 0;
    int32_t i = 0, n = 0, v;

    for (int32_t formatID = 0; formatID < 4; ++formatID) {
        while (sscanf(&str[i], formats[formatID], &v, &n) == 1 && n > 0) {
            int32_t vt = 0;

            if (sscanf(&str[i], "%d/%d", &v, &vt) == 1) {
                vt = 1;
            }
            CC_ASSERT(v >= 1 && vt >= 1 && "cc: unsupported relative index");

            if (halfedges != NULL) {
                (*halfedges)->twinID = -1;
                (*halfedges)->edgeID = -1;
                (*halfedges)->vertexID = v - 1;
                (*halfedges)->uvID = vt - 1;
                (*halfedges)+= 1;
            }

            i+= n;
            ++halfedgeCount;
        }
    }

    CC_ASSERT(halfedgeCount > 2);

    return halfedgeCount;
}


/*******************************************************************************
 * ObjReadVertex -- Reads an OBJ vertex
 *
 */
static int32_t ObjReadVertex(const char *buf, cc_VertexPoint **vertex)
{
    float *v = (*vertex)->array;

    if (sscanf(buf, "%f %f %f", &v[0], &v[1], &v[2]) == 3) {
        ++(*vertex);

        return 1;
    }

    return 0;
}


/*******************************************************************************
 * ObjReadUv -- Reads an OBJ texture coordinate
 *
 */
static int32_t ObjReadUv(const char *buf, cc_VertexUv **uv)
{
    float *vt = (*uv)->array;

    if (sscanf(buf, "%f %f", &vt[0], &vt[1]) == 2) {
        ++(*uv);

        return 1;
    }

    return 0;
}


/*******************************************************************************
 * ObjReadCrease -- Reads crease attribute
 *
 * This is a brute force approach: for each crease attribute, we iterate
 * over all half edges until we find those that should be sharpened.
 *
 */
static void
ObjReadCrease(
    cc_Mesh *mesh,
    const char *buffer
) {
    const int32_t creaseCount = ccm_CreaseCount(mesh);
    int32_t v0, v1;
    float s;

    if (sscanf(buffer, "t crease 2/1/0 %i %i %f", &v0, &v1, &s) == 3) {
        for (int32_t edgeID = 0; edgeID < creaseCount; ++edgeID) {
            const int32_t halfedgeID = ccm_EdgeToHalfedgeID(mesh, edgeID);
            const int32_t nextID = ccm_HalfedgeNextID(mesh, halfedgeID);
            const int32_t hv0 = ccm_HalfedgeVertexID(mesh, halfedgeID);
            const int32_t hv1 = ccm_HalfedgeVertexID(mesh, nextID);
            const bool b1 = (hv0 == v0) || (hv0 == v1);
            const bool b2 = (hv1 == v0) || (hv1 == v1);

            if (b1 && b2) {
                mesh->creases[edgeID].sharpness = s;
            }
        }
    }
}


/*******************************************************************************
 * ObjLoadMeshData -- Loads an OBJ mesh into a halfedge mesh
 *
 */
static bool
ObjLoadMeshData(FILE *stream, cc_Mesh *mesh, cbf_BitField *faceIterator)
{
    cc_Halfedge *halfedges = mesh->halfedges;
    cc_VertexPoint *vertexPoints = mesh->vertexPoints;
    cc_VertexUv *uvs = mesh->uvs;
    int32_t halfedgeCounter = 0;
    int32_t vertexCounter = 0;
    int32_t uvCounter = 0;
    char buffer[1024];

    cbf_SetBit(faceIterator, 0, 1u);

    while(fgets(buffer, sizeof(buffer), stream) != NULL) {
        if (buffer[0] == 'v') {
            if (buffer[1] == ' ') {
                vertexCounter+= ObjReadVertex(&buffer[2], &vertexPoints);
            } else if (buffer[1] == 't') {
                uvCounter+= ObjReadUv(&buffer[2], &uvs);
            }
        } else if (buffer[0] == 'f') {
            halfedgeCounter+= ObjReadFace(&buffer[2], &halfedges);
            cbf_SetBit(faceIterator, halfedgeCounter, 1u);
        }
    }

    if (halfedgeCounter != ccm_HalfedgeCount(mesh)
        || vertexCounter != ccm_VertexCount(mesh)
        || uvCounter != ccm_UvCount(mesh)) {
        return false;
    }

    cbf_Reduce(faceIterator);
    LoadFaceMappings(mesh, faceIterator);

    return true;
}


/*******************************************************************************
 * ObjLoadCreaseData -- Loads an OBJ crease mesh data into a halfedge mesh
 *
 */
static bool
ObjLoadCreaseData(FILE *stream, cc_Mesh *mesh)
{
    char buffer[1024];

    while(fgets(buffer, sizeof(buffer), stream) != NULL) {
        ObjReadCrease(mesh, buffer);
    }

    return true;
}


/*******************************************************************************
 * ObjReadMeshSize -- Retrieves the amount of memory suitable for loading an OBJ mesh
 *
 * This routine returns the number of indexes and vertices stored in the file.
 * Returns false if the size is invalid.
 *
 */
static bool
ObjReadMeshSize(
    FILE *stream,
    int32_t *halfedgeCount,
    int32_t *vertexCount,
    int32_t *uvCount
) {
    char buffer[1024];

    (*halfedgeCount) = 0;
    (*vertexCount) = 0;
    (*uvCount) = 0;

    while(fgets(buffer, sizeof(buffer), stream) != NULL) {
        if (buffer[0] == 'v') {
            if (buffer[1] == ' ') {
                ++(*vertexCount);
            } else if (buffer[1] == 't') {
                ++(*uvCount);
            }
        } else if (buffer[0] == 'f') {
            (*halfedgeCount)+= ObjReadFace(&buffer[1], NULL);
        }
    }

    return ((*halfedgeCount) > 0 && (*vertexCount) >= 3);
}


/*******************************************************************************
 * ObjReadVertex -- Reads an OBJ file
 *
 * Returns NULL on failure.
 *
 */
CCDEF cc_Mesh *LoadObj(const char *filename)
{
    int32_t halfedgeCount, vertexCount, uvCount;
    cbf_BitField *faceIterator;
    cc_Mesh *mesh;
    FILE *stream;

    stream = fopen(filename, "r");
    if (!stream) {
        CC_LOG("cc: fopen failed");

        return NULL;
    }

    if (!ObjReadMeshSize(stream, &halfedgeCount, &vertexCount, &uvCount)) {
        CC_LOG("cc: invalid OBJ file");
        fclose(stream);

        return NULL;
    }

    mesh = (cc_Mesh *)CC_MALLOC(sizeof(*mesh));
    mesh->halfedgeCount = halfedgeCount;
    mesh->halfedges = (cc_Halfedge *)CC_MALLOC(sizeof(cc_Halfedge) * halfedgeCount);
    mesh->vertexCount = vertexCount;
    mesh->vertexPoints = (cc_VertexPoint *)CC_MALLOC(sizeof(cc_VertexPoint) * vertexCount);
    mesh->uvCount = uvCount;
    mesh->uvs = (cc_VertexUv *)CC_MALLOC(sizeof(cc_VertexUv) * uvCount);
    faceIterator = cbf_Create(halfedgeCount + 1);
    rewind(stream);

    if (!ObjLoadMeshData(stream, mesh, faceIterator)) {
        CC_LOG("cc: failed to read OBJ data");
        CC_FREE(mesh->halfedges);
        CC_FREE(mesh->vertexPoints);
        CC_FREE(mesh->uvs);
        CC_FREE(mesh);

        fclose(stream);

        return NULL;
    }

    ComputeTwins(mesh);
    LoadEdgeMappings(mesh);
    LoadVertexHalfedges(mesh);

    if (true) {
        const int32_t creaseCount = ccm_EdgeCount(mesh);

        mesh->creases = (cc_Crease *)CC_MALLOC(sizeof(cc_Crease) * creaseCount);
        rewind(stream);

CC_PARALLEL_FOR
        for (int32_t creaseID = 0; creaseID < creaseCount; ++creaseID) {
            mesh->creases[creaseID].nextID = creaseID;
            mesh->creases[creaseID].prevID = creaseID;
            mesh->creases[creaseID].sharpness = 0.0f;
        }
CC_BARRIER

        if (!ObjLoadCreaseData(stream, mesh)) {
            CC_LOG("cc: failed to read OBJ crease data");
            ccm_Release(mesh);
            fclose(stream);

            return NULL;
        }

        MakeBoundariesSharp(mesh);
    }

    fclose(stream);
    cbf_Release(faceIterator);

    ComputeCreaseNeighbors(mesh);

    return mesh;
}

static void Usage(const char *appname)
{
    CC_LOG("usage -- %s file1 file2 ...", appname);
}


int main(int argc, char **argv)
{
    const int32_t meshCount = argc - 1;
    char buffer[1024];

    if (meshCount == 0) {
        Usage(argv[0]);
    }

    for (int32_t meshID = 0; meshID < meshCount; ++meshID) {
        const char *file = argv[meshID + 1];
        CC_LOG("Loading: %s", file);
        cc_Mesh *mesh = LoadObj(file);
        char *preFix, *postFix;

        memset(buffer, 0, sizeof(buffer));
        memcpy(buffer, file, strlen(file));
        preFix = strrchr(buffer, '/');
        postFix = strrchr(buffer, '.');

        if (postFix != NULL) {
            *postFix = '\0';
        }

        if (preFix != NULL) {
            sprintf(buffer, "%s.ccm", preFix + 1);
        } else {
            sprintf(buffer, "%s.ccm", buffer);
        }
        CC_LOG("Output file: %s", buffer);

        ccm_Save(mesh, buffer);
        ccm_Release(mesh);

        mesh = ccm_Load(buffer);
        CC_LOG("V: %i", ccm_VertexCount(mesh));
        CC_LOG("U: %i", ccm_UvCount(mesh));
        CC_LOG("H: %i", ccm_HalfedgeCount(mesh));
        CC_LOG("C: %i", ccm_CreaseCount(mesh));
        CC_LOG("E: %i", ccm_EdgeCount(mesh));
        CC_LOG("F: %i", ccm_FaceCount(mesh));
        ccm_Release(mesh);
    }
}
