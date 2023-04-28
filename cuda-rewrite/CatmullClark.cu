#include <omp.h>
#include <cuda.h>
#include <iostream>
#include "Utilities.h"
#include "Mesh.h"
#include "CatmullClark.h"

#define NUM_THREADS 256
#define TID (threadIdx.x + blockIdx.x * blockDim.x)
#define CHECK_TID(count) if (TID >= count) return;
#define EACH_ELEM(num_elems) (num_elems + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS

__device__ int32_t
ccs_VertexPointToHalfedgeID(const cc_Subd *subd, int32_t vertexID, int32_t depth)
{
#if 0 // recursive version
    if (depth > 1) {
        const cc_Mesh *cage = subd->cage;
        const int32_t vertexCount = ccm_VertexCountAtDepth_Fast(cage, depth - 1);
        const int32_t faceCount = ccm_FaceCountAtDepth_Fast(cage, depth - 1);

        if /* [V + F, V + F + E) */ (vertexID >= vertexCount + faceCount) {
            const int32_t edgeID = vertexID - vertexCount - faceCount;

            return 4 * ccs_EdgeToHalfedgeID(subd, edgeID, depth - 1) + 1;

        } else if /* [V, V + F) */ (vertexID >= vertexCount) {
            const int32_t faceID = vertexID - vertexCount;

            return 4 * ccm_FaceToHalfedgeID_Quad(faceID) + 2;

        } else /* [0, V) */ {

            return 4 * ccs_VertexPointToHalfedgeID(subd, vertexID, depth - 1) + 0;
        }
    } else {

        return ccs__VertexToHalfedgeID_First(subd->cage, vertexID);
    }
#else // non-recursive version
    const cc_Mesh *cage = subd->cage;
    int32_t heapDepth = depth;
    int32_t stride = 0;
    int32_t halfedgeID;

    // build heap
    for (; heapDepth > 1; --heapDepth) {
        const int32_t vertexCount = ccm_VertexCountAtDepth_Fast(cage, heapDepth - 1);
        const int32_t faceCount = ccm_FaceCountAtDepth_Fast(cage, heapDepth - 1);

        if /* [V + F, V + F + E) */ (vertexID >= vertexCount + faceCount) {
            const int32_t edgeID = vertexID - faceCount - vertexCount;

            halfedgeID = 4 * ccs_EdgeToHalfedgeID(subd, edgeID, heapDepth - 1) + 1;
            break;
        } else if /* [V, V + F) */ (vertexID >= vertexCount) {
            const int32_t faceID = vertexID - vertexCount;

            halfedgeID = 4 * ccm_FaceToHalfedgeID_Quad(faceID) + 2;
            break;
        } else /* [0, V) */ {
            stride+= 2;
        }
    }

    // initialize root cfg
    if (heapDepth == 1) {
        halfedgeID = ccs__VertexToHalfedgeID_First(subd->cage, vertexID);
    }

    return halfedgeID << stride;
#endif
}



/*******************************************************************************
 * CageFacePoints -- Applies Catmull Clark's face rule on the cage mesh
 *
 * The "Gather" routine iterates over each face of the mesh and compute the
 * resulting face vertex.
 *
 * The "Scatter" routine iterates over each halfedge of the mesh and atomically
 * adds its contribution to the computation of the face vertex.
 *
 */
__global__ void ccs__CageFacePoints_Gather(cc_Subd *subd)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t vertexCount = ccm_VertexCount(cage);
    const int32_t faceCount = ccm_FaceCount(cage);
    cc_VertexPoint *newFacePoints = &subd->vertexPoints[vertexCount];

    int edges_per_thread = std::ceil(float(faceCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t faceID = start; faceID < faceCount && faceID < end; ++faceID) {
        const int32_t halfedgeID = ccm_FaceToHalfedgeID(cage, faceID);
        cc_VertexPoint newFacePoint = ccm_HalfedgeVertexPoint(cage, halfedgeID);
        float faceVertexCount = 1.0f;

        for (int32_t halfedgeIt = ccm_HalfedgeNextID(cage, halfedgeID);
                     halfedgeIt != halfedgeID;
                     halfedgeIt = ccm_HalfedgeNextID(cage, halfedgeIt)) {
            const cc_VertexPoint vertexPoint = ccm_HalfedgeVertexPoint(cage, halfedgeIt);

            cc__Add3f(newFacePoint.array, newFacePoint.array, vertexPoint.array);
            ++faceVertexCount;
        }

        cc__Mul3f(newFacePoint.array, newFacePoint.array, 1.0f / faceVertexCount);

        newFacePoints[faceID] = newFacePoint;
    }

}

__global__ void ccs__CageFacePoints_Scatter(cc_Subd *subd)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t vertexCount = ccm_VertexCount(cage);
    const int32_t halfedgeCount = ccm_HalfedgeCount(cage);
    cc_VertexPoint *newFacePoints = &subd->vertexPoints[vertexCount];

int edges_per_thread = std::ceil(float(halfedgeCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t halfedgeID = start; halfedgeID < halfedgeCount && halfedgeID < end; ++halfedgeID) {
        const cc_VertexPoint vertexPoint = ccm_HalfedgeVertexPoint(cage, halfedgeID);
        const int32_t faceID = ccm_HalfedgeFaceID(cage, halfedgeID);
        float faceVertexCount = 1.0f;
        float *newFacePoint = newFacePoints[faceID].array;

        for (int32_t halfedgeIt = ccm_HalfedgeNextID(cage, halfedgeID);
                     halfedgeIt != halfedgeID;
                     halfedgeIt = ccm_HalfedgeNextID(cage, halfedgeIt)) {
            ++faceVertexCount;
        }

        for (int32_t i = 0; i < 3; ++i) {
// #pragma omp atomic
            // newFacePoint[i]+= vertexPoint.array[i] / (float)faceVertexCount;
            atomicAdd(newFacePoint + i, vertexPoint.array[i] / (float)faceVertexCount);
        }
    }

}


/*******************************************************************************
 * CageEdgePoints -- Applies Catmull Clark's edge rule on the cage mesh
 *
 * The "Gather" routine iterates over each edge of the mesh and computes the
 * resulting edge vertex.
 *
 * The "Scatter" routine iterates over each halfedge of the mesh and atomically
 * adds its contribution to the computation of the edge vertex.
 *
 */
__global__ void ccs__CageEdgePoints_Gather(cc_Subd *subd)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t vertexCount = ccm_VertexCount(cage);
    const int32_t edgeCount = ccm_EdgeCount(cage);
    const int32_t faceCount = ccm_FaceCount(cage);
    const cc_VertexPoint *newFacePoints = &subd->vertexPoints[vertexCount];
    cc_VertexPoint *newEdgePoints = &subd->vertexPoints[vertexCount + faceCount];

    int edges_per_thread = std::ceil(float(edgeCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t edgeID = start; edgeID < edgeCount && edgeID < end; ++edgeID) {
        const int32_t halfedgeID = ccm_EdgeToHalfedgeID(cage, edgeID);
        const int32_t twinID = ccm_HalfedgeTwinID(cage, halfedgeID);
        const int32_t nextID = ccm_HalfedgeNextID(cage, halfedgeID);
        const float edgeWeight = twinID < 0 ? 0.0f : 1.0f;
        const cc_VertexPoint oldEdgePoints[2] = {
            ccm_HalfedgeVertexPoint(cage, halfedgeID),
            ccm_HalfedgeVertexPoint(cage,     nextID)
        };
        const cc_VertexPoint newFacePointPair[2] = {
            newFacePoints[ccm_HalfedgeFaceID(cage, halfedgeID)],
            newFacePoints[ccm_HalfedgeFaceID(cage, cc__Max(0, twinID))]
        };
        float *newEdgePoint = newEdgePoints[edgeID].array;
        cc_VertexPoint sharpEdgePoint = {0.0f, 0.0f, 0.0f};
        cc_VertexPoint smoothEdgePoint = {0.0f, 0.0f, 0.0f};
        float tmp1[3], tmp2[3];

        cc__Add3f(tmp1, oldEdgePoints[0].array, oldEdgePoints[1].array);
        cc__Add3f(tmp2, newFacePointPair[0].array, newFacePointPair[1].array);
        cc__Mul3f(sharpEdgePoint.array, tmp1, 0.5f);
        cc__Add3f(smoothEdgePoint.array, tmp1, tmp2);
        cc__Mul3f(smoothEdgePoint.array, smoothEdgePoint.array, 0.25f);
        cc__Lerp3f(newEdgePoint,
                   sharpEdgePoint.array,
                   smoothEdgePoint.array,
                   edgeWeight);
    }
    __syncthreads();

}

__global__ void ccs__CageEdgePoints_Scatter(cc_Subd *subd)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t faceCount = ccm_FaceCount(cage);
    const int32_t vertexCount = ccm_VertexCount(cage);
    const int32_t halfedgeCount = ccm_HalfedgeCount(cage);
    const cc_VertexPoint *newFacePoints = &subd->vertexPoints[vertexCount];
    cc_VertexPoint *newEdgePoints = &subd->vertexPoints[vertexCount + faceCount];

    int edges_per_thread = std::ceil(float(halfedgeCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t halfedgeID = start; halfedgeID < halfedgeCount && halfedgeID < end; ++halfedgeID) {
        const int32_t faceID = ccm_HalfedgeFaceID(cage, halfedgeID);
        const int32_t edgeID = ccm_HalfedgeEdgeID(cage, halfedgeID);
        const int32_t twinID = ccm_HalfedgeTwinID(cage, halfedgeID);
        const int32_t nextID = ccm_HalfedgeNextID(cage, halfedgeID);
        const cc_VertexPoint newFacePoint = newFacePoints[faceID];
        float tmp1[3], tmp2[3], tmp3[3], tmp4[3], atomicWeight[3];
        float weight = twinID >= 0 ? 0.5f : 1.0f;

        cc__Mul3f(tmp1, newFacePoint.array, 0.5f);
        cc__Mul3f(tmp2, ccm_HalfedgeVertexPoint(cage, halfedgeID).array, weight);
        cc__Mul3f(tmp3, ccm_HalfedgeVertexPoint(cage,     nextID).array, weight);
        cc__Lerp3f(tmp4, tmp2, tmp3, 0.5f);
        cc__Lerp3f(atomicWeight, tmp1, tmp4, weight);

        for (int32_t i = 0; i < 3; ++i) {
// #pragma omp atomic
            // newEdgePoints[edgeID].array[i]+= atomicWeight[i];
            atomicAdd(newEdgePoints[edgeID].array + i, atomicWeight[i]);
        }
    }
    __syncthreads();

}


/*******************************************************************************
 * CreasedCageEdgePoints -- Applies DeRole et al.'s edge rule on the cage mesh
 *
 * The "Gather" routine iterates over each edge of the mesh and computes the
 * resulting edge vertex.
 *
 * The "Scatter" routine iterates over each halfedge of the mesh and atomically
 * adds its contribution to the computation of the edge vertex.
 *
 */
__global__ void ccs__CreasedCageEdgePoints_Gather(cc_Subd *subd)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t vertexCount = ccm_VertexCount(cage);
    const int32_t edgeCount = ccm_EdgeCount(cage);
    const int32_t faceCount = ccm_FaceCount(cage);
    cc_VertexPoint *newFacePoints = &subd->vertexPoints[vertexCount];
    cc_VertexPoint *newEdgePoints = &subd->vertexPoints[vertexCount + faceCount];

int edges_per_thread = std::ceil(float(edgeCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t edgeID = start; edgeID < edgeCount && edgeID < end; ++edgeID) {
        const int32_t halfedgeID = ccm_EdgeToHalfedgeID(cage, edgeID);
        const int32_t twinID = ccm_HalfedgeTwinID(cage, halfedgeID);
        const int32_t nextID = ccm_HalfedgeNextID(cage, halfedgeID);
        const float sharp = ccm_CreaseSharpness(cage, edgeID);
        const float edgeWeight = cc__Satf(sharp);
        const cc_VertexPoint oldEdgePoints[2] = {
            ccm_HalfedgeVertexPoint(cage, halfedgeID),
            ccm_HalfedgeVertexPoint(cage,     nextID)
        };
        const cc_VertexPoint newAdjacentFacePoints[2] = {
            newFacePoints[ccm_HalfedgeFaceID(cage, halfedgeID)],
            newFacePoints[ccm_HalfedgeFaceID(cage, cc__Max(0, twinID))]
        };
        cc_VertexPoint sharpEdgePoint = {0.0f, 0.0f, 0.0f};
        cc_VertexPoint smoothEdgePoint = {0.0f, 0.0f, 0.0f};
        float tmp1[3], tmp2[3];

        cc__Add3f(tmp1, oldEdgePoints[0].array, oldEdgePoints[1].array);
        cc__Add3f(tmp2, newAdjacentFacePoints[0].array, newAdjacentFacePoints[1].array);
        cc__Mul3f(sharpEdgePoint.array, tmp1, 0.5f);
        cc__Add3f(smoothEdgePoint.array, tmp1, tmp2);
        cc__Mul3f(smoothEdgePoint.array, smoothEdgePoint.array, 0.25f);
        cc__Lerp3f(newEdgePoints[edgeID].array,
                   smoothEdgePoint.array,
                   sharpEdgePoint.array,
                   edgeWeight);
    }
    __syncthreads();

}

__global__ void ccs__CreasedCageEdgePoints_Scatter(cc_Subd *subd)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t faceCount = ccm_FaceCount(cage);
    const int32_t vertexCount = ccm_VertexCount(cage);
    const int32_t halfedgeCount = ccm_HalfedgeCount(cage);
    const cc_VertexPoint *newFacePoints = &subd->vertexPoints[vertexCount];
    cc_VertexPoint *newEdgePoints = &subd->vertexPoints[vertexCount + faceCount];

int edges_per_thread = std::ceil(float(halfedgeCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t halfedgeID = start; halfedgeID < halfedgeCount && halfedgeID < end; ++halfedgeID) {
        const int32_t faceID = ccm_HalfedgeFaceID(cage, halfedgeID);
        const int32_t edgeID = ccm_HalfedgeEdgeID(cage, halfedgeID);
        const int32_t twinID = ccm_HalfedgeTwinID(cage, halfedgeID);
        const int32_t nextID = ccm_HalfedgeNextID(cage, halfedgeID);
        const float sharp = ccm_CreaseSharpness(cage, edgeID);
        const float edgeWeight = cc__Satf(sharp);
        const cc_VertexPoint newFacePoint = newFacePoints[faceID];
        const cc_VertexPoint oldEdgePoints[2] = {
            ccm_HalfedgeVertexPoint(cage, halfedgeID),
            ccm_HalfedgeVertexPoint(cage,     nextID)
        };
        cc_VertexPoint smoothPoint = {0.0f, 0.0f, 0.0f};
        cc_VertexPoint sharpPoint = {0.0f, 0.0f, 0.0f};
        float tmp[3], atomicWeight[3];

        // sharp point
        cc__Lerp3f(tmp, oldEdgePoints[0].array, oldEdgePoints[1].array, 0.5f);
        cc__Mul3f(sharpPoint.array, tmp, twinID < 0 ? 1.0f : 0.5f);

        // smooth point
        cc__Lerp3f(tmp, oldEdgePoints[0].array, newFacePoint.array, 0.5f);
        cc__Mul3f(smoothPoint.array, tmp, 0.5f);

        // atomic weight
        cc__Lerp3f(atomicWeight,
                   smoothPoint.array,
                   sharpPoint.array,
                   edgeWeight);

        for (int32_t i = 0; i < 3; ++i) {
// #pragma omp atomic
            // newEdgePoints[edgeID].array[i]+= atomicWeight[i];
            atomicAdd(newEdgePoints[edgeID].array + i, atomicWeight[i]);
        }
    }
    __syncthreads();
}


/*******************************************************************************
 * CageVertexPoints -- Applies Catmull Clark's vertex rule on the cage mesh
 *
 * The "Gather" routine iterates over each vertex of the mesh and computes the
 * resulting smooth vertex.
 *
 * The "Scatter" routine iterates over each halfedge of the mesh and atomically
 * adds its contribution to the computation of the smooth vertex.
 *
 */
__global__ void ccs__CageVertexPoints_Gather(cc_Subd *subd)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t vertexCount = ccm_VertexCount(cage);
    const int32_t faceCount = ccm_FaceCount(cage);
    const cc_VertexPoint *newFacePoints = &subd->vertexPoints[vertexCount];
    const cc_VertexPoint *newEdgePoints = &subd->vertexPoints[vertexCount + faceCount];
    cc_VertexPoint *newVertexPoints = subd->vertexPoints;

    int edges_per_thread = std::ceil(float(vertexCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t vertexID = start; vertexID < vertexCount && vertexID < end; ++vertexID) {
        const int32_t halfedgeID = ccm_VertexToHalfedgeID(cage, vertexID);
        const int32_t edgeID = ccm_HalfedgeEdgeID(cage, halfedgeID);
        const int32_t faceID = ccm_HalfedgeFaceID(cage, halfedgeID);
        const cc_VertexPoint newEdgePoint = newEdgePoints[edgeID];
        const cc_VertexPoint newFacePoint = newFacePoints[faceID];
        const cc_VertexPoint oldVertexPoint = ccm_VertexPoint(cage, vertexID);
        cc_VertexPoint smoothPoint = {0.0f, 0.0f, 0.0f};
        float valence = 1.0f;
        int32_t iterator;
        float tmp1[3], tmp2[3];

        cc__Mul3f(tmp1, newFacePoint.array, -1.0f);
        cc__Mul3f(tmp2, newEdgePoint.array, +4.0f);
        cc__Add3f(smoothPoint.array, tmp1, tmp2);

        for (iterator = ccm_PrevVertexHalfedgeID(cage, halfedgeID);
             iterator >= 0 && iterator != halfedgeID;
             iterator = ccm_PrevVertexHalfedgeID(cage, iterator)) {
            const int32_t edgeID = ccm_HalfedgeEdgeID(cage, iterator);
            const int32_t faceID = ccm_HalfedgeFaceID(cage, iterator);
            const cc_VertexPoint newEdgePoint = newEdgePoints[edgeID];
            const cc_VertexPoint newFacePoint = newFacePoints[faceID];

            cc__Mul3f(tmp1, newFacePoint.array, -1.0f);
            cc__Mul3f(tmp2, newEdgePoint.array, +4.0f);
            cc__Add3f(smoothPoint.array, smoothPoint.array, tmp1);
            cc__Add3f(smoothPoint.array, smoothPoint.array, tmp2);
            ++valence;
        }

        cc__Mul3f(tmp1, smoothPoint.array, 1.0f / (valence * valence));
        cc__Mul3f(tmp2, oldVertexPoint.array, 1.0f - 3.0f / valence);
        cc__Add3f(smoothPoint.array, tmp1, tmp2);
        cc__Lerp3f(newVertexPoints[vertexID].array,
                   oldVertexPoint.array,
                   smoothPoint.array,
                   iterator != halfedgeID ? 0.0f : 1.0f);
    }
    __syncthreads();
}

__global__ void ccs__CageVertexPoints_Scatter(cc_Subd *subd)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t faceCount = ccm_FaceCount(cage);
    const int32_t vertexCount = ccm_VertexCount(cage);
    const int32_t halfedgeCount = ccm_HalfedgeCount(cage);
    const cc_VertexPoint *newFacePoints = &subd->vertexPoints[vertexCount];
    const cc_VertexPoint *newEdgePoints = &subd->vertexPoints[vertexCount + faceCount];
    cc_VertexPoint *newVertexPoints = subd->vertexPoints;

    int edges_per_thread = std::ceil(float(halfedgeCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t halfedgeID = start; halfedgeID < halfedgeCount && halfedgeID < end; ++halfedgeID) {
        const int32_t vertexID = ccm_HalfedgeVertexID(cage, halfedgeID);
        const int32_t edgeID = ccm_HalfedgeEdgeID(cage, halfedgeID);
        const int32_t faceID = ccm_HalfedgeFaceID(cage, halfedgeID);
        const cc_VertexPoint oldVertexPoint = ccm_VertexPoint(cage, vertexID);
        int32_t valence = 1;
        int32_t forwardIterator, backwardIterator;

        for (forwardIterator = ccm_PrevVertexHalfedgeID(cage, halfedgeID);
             forwardIterator >= 0 && forwardIterator != halfedgeID;
             forwardIterator = ccm_PrevVertexHalfedgeID(cage, forwardIterator)) {
            ++valence;
        }

        for (backwardIterator = ccm_NextVertexHalfedgeID(cage, halfedgeID);
             forwardIterator < 0 && backwardIterator >= 0 && backwardIterator != halfedgeID;
             backwardIterator = ccm_NextVertexHalfedgeID(cage, backwardIterator)) {
            ++valence;
        }

        for (int32_t i = 0; i < 3; ++i) {
            const float w = 1.0f / (float)valence;
            const float v = oldVertexPoint.array[i];
            const float f = newFacePoints[faceID].array[i];
            const float e = newEdgePoints[edgeID].array[i];
            const float s = forwardIterator < 0 ? 0.0f : 1.0f;
// #pragma omp atomic
            // newVertexPoints[vertexID].array[i]+=
                // w * (v + w * s * (4.0f * e - f - 3.0f * v));
            atomicAdd(newVertexPoints[vertexID].array + i, w * (v + w * s * (4.0f * e - f - 3.0f * v)));
        }
    }
    __syncthreads();
}


/*******************************************************************************
 * CreasedCageVertexPoints -- Applies DeRose et al.'s vertex rule on cage mesh
 *
 * The "Gather" routine iterates over each vertex of the mesh and computes the
 * resulting smooth vertex.
 *
 * The "Scatter" routine iterates over each halfedge of the mesh and atomically
 * adds its contribution to the computation of the smooth vertex.
 *
 */
__global__ void ccs__CreasedCageVertexPoints_Gather(cc_Subd *subd)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t vertexCount = ccm_VertexCount(cage);
    const int32_t faceCount = ccm_FaceCount(cage);
    const cc_VertexPoint *newFacePoints = &subd->vertexPoints[vertexCount];
    const cc_VertexPoint *newEdgePoints = &subd->vertexPoints[vertexCount + faceCount];
    cc_VertexPoint *newVertexPoints = subd->vertexPoints;

    int edges_per_thread = std::ceil(float(vertexCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t vertexID = start; vertexID < vertexCount && vertexID < end; ++vertexID) {
        const int32_t halfedgeID = ccm_VertexToHalfedgeID(cage, vertexID);
        const int32_t edgeID = ccm_HalfedgeEdgeID(cage, halfedgeID);
        const int32_t prevID = ccm_HalfedgePrevID(cage, halfedgeID);
        const int32_t prevEdgeID = ccm_HalfedgeEdgeID(cage, prevID);
        const int32_t prevFaceID = ccm_HalfedgeFaceID(cage, prevID);
        const float thisS = ccm_HalfedgeSharpness(cage, halfedgeID);
        const float prevS = ccm_HalfedgeSharpness(cage,     prevID);
        const float creaseWeight = cc__Signf(thisS);
        const float prevCreaseWeight = cc__Signf(prevS);
        const cc_VertexPoint newEdgePoint = newEdgePoints[edgeID];
        const cc_VertexPoint newPrevEdgePoint = newEdgePoints[prevEdgeID];
        const cc_VertexPoint newPrevFacePoint = newFacePoints[prevFaceID];
        const cc_VertexPoint oldPoint = ccm_VertexPoint(cage, vertexID);
        cc_VertexPoint smoothPoint = {0.0f, 0.0f, 0.0f};
        cc_VertexPoint creasePoint = {0.0f, 0.0f, 0.0f};
        float avgS = prevS;
        float creaseCount = prevCreaseWeight;
        float valence = 1.0f;
        int32_t forwardIterator;
        float tmp1[3], tmp2[3];

        // smooth contrib
        cc__Mul3f(tmp1, newPrevFacePoint.array, -1.0f);
        cc__Mul3f(tmp2, newPrevEdgePoint.array, +4.0f);
        cc__Add3f(smoothPoint.array, tmp1, tmp2);

        // crease contrib
        cc__Mul3f(tmp1, newPrevEdgePoint.array, prevCreaseWeight);
        cc__Add3f(creasePoint.array, creasePoint.array, tmp1);

        for (forwardIterator = ccm_HalfedgeTwinID(cage, prevID);
             forwardIterator >= 0 && forwardIterator != halfedgeID;
             forwardIterator = ccm_HalfedgeTwinID(cage, forwardIterator)) {
            const int32_t prevID = ccm_HalfedgePrevID(cage, forwardIterator);
            const int32_t prevEdgeID = ccm_HalfedgeEdgeID(cage, prevID);
            const int32_t prevFaceID = ccm_HalfedgeFaceID(cage, prevID);
            const cc_VertexPoint newPrevEdgePoint = newEdgePoints[prevEdgeID];
            const cc_VertexPoint newPrevFacePoint = newFacePoints[prevFaceID];
            const float prevS = ccm_HalfedgeSharpness(cage, prevID);
            const float prevCreaseWeight = cc__Signf(prevS);

            // smooth contrib
            cc__Mul3f(tmp1, newPrevFacePoint.array, -1.0f);
            cc__Mul3f(tmp2, newPrevEdgePoint.array, +4.0f);
            cc__Add3f(smoothPoint.array, smoothPoint.array, tmp1);
            cc__Add3f(smoothPoint.array, smoothPoint.array, tmp2);
            ++valence;

            // crease contrib
            cc__Mul3f(tmp1, newPrevEdgePoint.array, prevCreaseWeight);
            cc__Add3f(creasePoint.array, creasePoint.array, tmp1);
            avgS+= prevS;
            creaseCount+= prevCreaseWeight;

            // next vertex halfedge
            forwardIterator = prevID;
        }

        // boundary corrections
        if (forwardIterator < 0) {
            cc__Mul3f(tmp1, newEdgePoint.array    , creaseWeight);
            cc__Add3f(creasePoint.array, creasePoint.array, tmp1);
            creaseCount+= creaseWeight;
            ++valence;
        }

        // smooth point
        cc__Mul3f(tmp1, smoothPoint.array, 1.0f / (valence * valence));
        cc__Mul3f(tmp2, oldPoint.array, 1.0f - 3.0f / valence);
        cc__Add3f(smoothPoint.array, tmp1, tmp2);

        // crease point
        cc__Mul3f(tmp1, creasePoint.array, 0.25f);
        cc__Mul3f(tmp2, oldPoint.array, 0.5f);
        cc__Add3f(creasePoint.array, tmp1, tmp2);

        // proper vertex rule selection
        if (creaseCount <= 1.0f) {
            newVertexPoints[vertexID] = smoothPoint;
        } else if (creaseCount >= 3.0f || valence == 2.0f) {
            newVertexPoints[vertexID] = oldPoint;
        } else {
            cc__Lerp3f(newVertexPoints[vertexID].array,
                       oldPoint.array,
                       creasePoint.array,
                       cc__Satf(avgS * 0.5f));
        }
    }
    __syncthreads();
}


__global__ void ccs__CreasedCageVertexPoints_Scatter(cc_Subd *subd)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t faceCount = ccm_FaceCount(cage);
    const int32_t vertexCount = ccm_VertexCount(cage);
    const int32_t halfedgeCount = ccm_HalfedgeCount(cage);
    const cc_VertexPoint *oldVertexPoints = cage->vertexPoints;
    const cc_VertexPoint *newFacePoints = &subd->vertexPoints[vertexCount];
    const cc_VertexPoint *newEdgePoints = &subd->vertexPoints[vertexCount + faceCount];
    cc_VertexPoint *newVertexPoints = subd->vertexPoints;

    int edges_per_thread = std::ceil(float(halfedgeCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t halfedgeID = start; halfedgeID < halfedgeCount && halfedgeID < end; ++halfedgeID) {
        const int32_t vertexID = ccm_HalfedgeVertexID(cage, halfedgeID);
        const int32_t edgeID = ccm_HalfedgeEdgeID(cage, halfedgeID);
        const int32_t faceID = ccm_HalfedgeFaceID(cage, halfedgeID);
        const int32_t prevID = ccm_HalfedgePrevID(cage, halfedgeID);
        const int32_t prevEdgeID = ccm_HalfedgeEdgeID(cage, prevID);
        const float thisS = ccm_HalfedgeSharpness(cage, halfedgeID);
        const float prevS = ccm_HalfedgeSharpness(cage,     prevID);
        const float creaseWeight = cc__Signf(thisS);
        const float prevCreaseWeight = cc__Signf(prevS);
        const cc_VertexPoint newPrevEdgePoint = newEdgePoints[prevEdgeID];
        const cc_VertexPoint newEdgePoint = newEdgePoints[edgeID];
        const cc_VertexPoint newFacePoint = newFacePoints[faceID];
        const cc_VertexPoint oldPoint = oldVertexPoints[vertexID];
        cc_VertexPoint cornerPoint = {0.0f, 0.0f, 0.0f};
        cc_VertexPoint smoothPoint = {0.0f, 0.0f, 0.0f};
        cc_VertexPoint creasePoint = {0.0f, 0.0f, 0.0f};
        cc_VertexPoint atomicWeight = {0.0f, 0.0f, 0.0f};
        float avgS = prevS;
        float creaseCount = prevCreaseWeight;
        float valence = 1.0f;
        int32_t forwardIterator, backwardIterator;
        float tmp1[3], tmp2[3];

        for (forwardIterator = ccm_HalfedgeTwinID(cage, prevID);
             forwardIterator >= 0 && forwardIterator != halfedgeID;
             forwardIterator = ccm_HalfedgeTwinID(cage, forwardIterator)) {
            const int32_t prevID = ccm_HalfedgePrevID(cage, forwardIterator);
            const float prevS = ccm_HalfedgeSharpness(cage, prevID);
            const float prevCreaseWeight = cc__Signf(prevS);

            // valence computation
            ++valence;

            // crease computation
            avgS+= prevS;
            creaseCount+= prevCreaseWeight;

            // next vertex halfedge
            forwardIterator = prevID;
        }

        for (backwardIterator = ccm_HalfedgeTwinID(cage, halfedgeID);
             forwardIterator < 0 && backwardIterator >= 0 && backwardIterator != halfedgeID;
             backwardIterator = ccm_HalfedgeTwinID(cage, backwardIterator)) {
            const int32_t nextID = ccm_HalfedgeNextID(cage, backwardIterator);
            const float nextS = ccm_HalfedgeSharpness(cage, nextID);
            const float nextCreaseWeight = cc__Signf(nextS);

            // valence computation
            ++valence;

            // crease computation
            avgS+= nextS;
            creaseCount+= nextCreaseWeight;

            // next vertex halfedge
            backwardIterator = nextID;
        }

        // corner point
        cc__Mul3f(cornerPoint.array, oldPoint.array, 1.0f / valence);

        // crease computation: V / 4
        cc__Mul3f(tmp1, oldPoint.array, 0.25f * creaseWeight);
        cc__Mul3f(tmp2, newEdgePoint.array, 0.25f * creaseWeight);
        cc__Add3f(creasePoint.array, tmp1, tmp2);

        // smooth computation: (4E - F + (n - 3) V) / N
        cc__Mul3f(tmp1, newFacePoint.array, -1.0f);
        cc__Mul3f(tmp2, newEdgePoint.array, +4.0f);
        cc__Add3f(smoothPoint.array, tmp1, tmp2);
        cc__Mul3f(tmp1, oldPoint.array, valence - 3.0f);
        cc__Add3f(smoothPoint.array, smoothPoint.array, tmp1);
        cc__Mul3f(smoothPoint.array,
                  smoothPoint.array,
                  1.0f / (valence * valence));

        // boundary corrections
        if (forwardIterator < 0) {
            creaseCount+= creaseWeight;
            ++valence;

            cc__Mul3f(tmp1, oldPoint.array, 0.25f * prevCreaseWeight);
            cc__Mul3f(tmp2, newPrevEdgePoint.array, 0.25f * prevCreaseWeight);
            cc__Add3f(tmp1, tmp1, tmp2);
            cc__Add3f(creasePoint.array, creasePoint.array, tmp1);
        }

        // atomicWeight (TODO: make branchless ?)
        if (creaseCount <= 1.0f) {
            atomicWeight = smoothPoint;
        } else if (creaseCount >= 3.0f || valence == 2.0f) {
            atomicWeight = cornerPoint;
        } else {
            cc__Lerp3f(atomicWeight.array,
                       cornerPoint.array,
                       creasePoint.array,
                       cc__Satf(avgS * 0.5f));
        }
        if(halfedgeID == 0){
            printf("atomicWeight (%02f, %02f, %02f)\n", atomicWeight.array[0], atomicWeight.array[1], atomicWeight.array[2]);
        }
        for (int32_t i = 0; i < 3; ++i) {
// #pragma omp atomic
            // newVertexPoints[vertexID].array[i]+= atomicWeight.array[i];
            atomicAdd(&newVertexPoints[vertexID].array[i], atomicWeight.array[i]);
            
        }
        if(halfedgeID == 0){
            printf("newVertexPoints (%02f, %02f, %02f)\n", newVertexPoints[vertexID].array[0], newVertexPoints[vertexID].array[1], newVertexPoints[vertexID].array[2]);
        }
    }
    __syncthreads();
}


/*******************************************************************************
 * FacePoints -- Applies Catmull Clark's face rule on the subd
 *
 * The "Gather" routine iterates over each face of the mesh and compute the
 * resulting face vertex.
 *
 * The "Scatter" routine iterates over each halfedge of the mesh and atomically
 * adds its contribution to the computation of the face vertex.
 *
 */
__global__ void ccs__FacePoints_Gather(cc_Subd *subd, int32_t depth)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t vertexCount = ccm_VertexCountAtDepth_Fast(cage, depth);
    const int32_t faceCount = ccm_FaceCountAtDepth_Fast(cage, depth);
    const int32_t stride = ccs_CumulativeVertexCountAtDepth(cage, depth);
    cc_VertexPoint *newFacePoints = &subd->vertexPoints[stride + vertexCount];

    int edges_per_thread = std::ceil(float(faceCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t faceID = start; faceID < faceCount && faceID < end; ++faceID) {
        const int32_t halfedgeID = ccs_FaceToHalfedgeID(subd, faceID, depth);
        cc_VertexPoint newFacePoint = ccs_HalfedgeVertexPoint(subd, halfedgeID, depth);

        for (int32_t halfedgeIt = ccs_HalfedgeNextID(subd, halfedgeID, depth);
                     halfedgeIt != halfedgeID;
                     halfedgeIt = ccs_HalfedgeNextID(subd, halfedgeIt, depth)) {
            const cc_VertexPoint vertexPoint = ccs_HalfedgeVertexPoint(subd, halfedgeIt, depth);

            cc__Add3f(newFacePoint.array, newFacePoint.array, vertexPoint.array);
        }

        cc__Mul3f(newFacePoint.array, newFacePoint.array, 0.25f);

        newFacePoints[faceID] = newFacePoint;
    }
    __syncthreads();
}

__global__ void ccs__FacePoints_Scatter(cc_Subd *subd, int32_t depth)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t halfedgeCount = ccm_HalfedgeCountAtDepth(cage, depth);
    const int32_t vertexCount = ccm_VertexCountAtDepth_Fast(cage, depth);
    const int32_t stride = ccs_CumulativeVertexCountAtDepth(cage, depth);
    cc_VertexPoint *newFacePoints = &subd->vertexPoints[stride + vertexCount];

    int edges_per_thread = std::ceil(float(halfedgeCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t halfedgeID = start; halfedgeID < halfedgeCount && halfedgeID < end; ++halfedgeID) {
        const cc_VertexPoint vertexPoint = ccs_HalfedgeVertexPoint(subd, halfedgeID, depth);
        const int32_t faceID = ccs_HalfedgeFaceID(subd, halfedgeID, depth);
        float *newFacePoint = newFacePoints[faceID].array;

        for (int32_t i = 0; i < 3; ++i) {
            atomicAdd(&newFacePoint[i], vertexPoint.array[i] / (float)4.0f);
            // newFacePoint[i]+= vertexPoint.array[i] / (float)4.0f;
        }
    }
    __syncthreads();
}


/*******************************************************************************
 * EdgePoints -- Applies Catmull Clark's edge rule on the subd
 *
 * The "Gather" routine iterates over each edge of the mesh and compute the
 * resulting edge vertex.
 *
 * The "Scatter" routine iterates over each halfedge of the mesh and atomically
 * adds its contribution to the computation of the edge vertex.
 *
 */
__global__ void ccs__EdgePoints_Gather(cc_Subd *subd, int32_t depth)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t vertexCount = ccm_VertexCountAtDepth_Fast(cage, depth);
    const int32_t edgeCount = ccm_EdgeCountAtDepth_Fast(cage, depth);
    const int32_t faceCount = ccm_FaceCountAtDepth_Fast(cage, depth);
    const int32_t stride = ccs_CumulativeVertexCountAtDepth(cage, depth);
    const cc_VertexPoint *newFacePoints = &subd->vertexPoints[stride + vertexCount];
    cc_VertexPoint *newEdgePoints = &subd->vertexPoints[stride + vertexCount + faceCount];

    int edges_per_thread = std::ceil(float(edgeCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t edgeID = start; edgeID < edgeCount && edgeID < end; ++edgeID) {
        const int32_t halfedgeID = ccs_EdgeToHalfedgeID(subd, edgeID, depth);
        const int32_t twinID = ccs_HalfedgeTwinID(subd, halfedgeID, depth);
        const int32_t nextID = ccs_HalfedgeNextID(subd, halfedgeID, depth);
        const float edgeWeight = twinID < 0 ? 0.0f : 1.0f;
        const cc_VertexPoint oldEdgePoints[2] = {
            ccs_HalfedgeVertexPoint(subd, halfedgeID, depth),
            ccs_HalfedgeVertexPoint(subd,     nextID, depth)
        };
        const cc_VertexPoint newAdjacentFacePoints[2] = {
            newFacePoints[ccs_HalfedgeFaceID(subd,         halfedgeID, depth)],
            newFacePoints[ccs_HalfedgeFaceID(subd, cc__Max(0, twinID), depth)]
        };
        float *newEdgePoint = newEdgePoints[edgeID].array;
        cc_VertexPoint sharpEdgePoint = {0.0f, 0.0f, 0.0f};
        cc_VertexPoint smoothEdgePoint = {0.0f, 0.0f, 0.0f};
        float tmp1[3], tmp2[3];

        cc__Add3f(tmp1, oldEdgePoints[0].array, oldEdgePoints[1].array);
        cc__Add3f(tmp2, newAdjacentFacePoints[0].array, newAdjacentFacePoints[1].array);
        cc__Mul3f(sharpEdgePoint.array, tmp1, 0.5f);
        cc__Add3f(smoothEdgePoint.array, tmp1, tmp2);
        cc__Mul3f(smoothEdgePoint.array, smoothEdgePoint.array, 0.25f);
        cc__Lerp3f(newEdgePoint,
                   sharpEdgePoint.array,
                   smoothEdgePoint.array,
                   edgeWeight);
    }
    __syncthreads();
}

__global__ void ccs__EdgePoints_Scatter(cc_Subd *subd, int32_t depth)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t halfedgeCount = ccm_HalfedgeCountAtDepth(cage, depth);
    const int32_t vertexCount = ccm_VertexCountAtDepth_Fast(cage, depth);
    const int32_t faceCount = ccm_FaceCountAtDepth_Fast(cage, depth);
    const int32_t stride = ccs_CumulativeVertexCountAtDepth(cage, depth);
    const cc_VertexPoint *newFacePoints = &subd->vertexPoints[stride + vertexCount];
    cc_VertexPoint *newEdgePoints = &subd->vertexPoints[stride + vertexCount + faceCount];

    int edges_per_thread = std::ceil(float(halfedgeCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t halfedgeID = start; halfedgeID < halfedgeCount && halfedgeID < end; ++halfedgeID) {
        const int32_t faceID = ccs_HalfedgeFaceID(subd, halfedgeID, depth);
        const int32_t edgeID = ccs_HalfedgeEdgeID(subd, halfedgeID, depth);
        const int32_t twinID = ccs_HalfedgeTwinID(subd, halfedgeID, depth);
        const int32_t nextID = ccs_HalfedgeNextID(subd, halfedgeID, depth);
        const cc_VertexPoint newFacePoint = newFacePoints[faceID];
        float tmp1[3], tmp2[3], tmp3[3], tmp4[3], atomicWeight[3];
        float weight = twinID >= 0 ? 0.5f : 1.0f;

        cc__Mul3f(tmp1, newFacePoint.array, 0.5f);
        cc__Mul3f(tmp2, ccs_HalfedgeVertexPoint(subd, halfedgeID, depth).array, weight);
        cc__Mul3f(tmp3, ccs_HalfedgeVertexPoint(subd,     nextID, depth).array, weight);
        cc__Lerp3f(tmp4, tmp2, tmp3, 0.5f);
        cc__Lerp3f(atomicWeight, tmp1, tmp4, weight);

        for (int32_t i = 0; i < 3; ++i) {
    // #pragma omp atomic
            // newEdgePoints[edgeID].array[i]+= atomicWeight[i];
            atomicAdd(newEdgePoints[edgeID].array + i, atomicWeight[i]);
        }
    }
    __syncthreads();
}

/*******************************************************************************
 * CreasedEdgePoints -- Applies DeRose et al's edge rule on the subd
 *
 * The "Gather" routine iterates over each edge of the mesh and compute the
 * resulting edge vertex.
 *
 * The "Scatter" routine iterates over each halfedge of the mesh and atomically
 * adds its contribution to the computation of the edge vertex.
 *
 */
__global__ void ccs__CreasedEdgePoints_Gather(cc_Subd *subd, int32_t depth)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t vertexCount = ccm_VertexCountAtDepth_Fast(cage, depth);
    const int32_t faceCount = ccm_FaceCountAtDepth_Fast(cage, depth);
    const int32_t edgeCount = ccm_EdgeCountAtDepth_Fast(cage, depth);
    const int32_t stride = ccs_CumulativeVertexCountAtDepth(cage, depth);
    const cc_VertexPoint *newFacePoints = &subd->vertexPoints[stride + vertexCount];
    cc_VertexPoint *newEdgePoints = &subd->vertexPoints[stride +vertexCount + faceCount];

    int edges_per_thread = std::ceil(float(edgeCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t edgeID = start; edgeID < edgeCount && edgeID < end; ++edgeID) {
        const int32_t halfedgeID = ccs_EdgeToHalfedgeID(subd, edgeID, depth);
        const int32_t twinID = ccs_HalfedgeTwinID(subd, halfedgeID, depth);
        const int32_t nextID = ccs_HalfedgeNextID(subd, halfedgeID, depth);
        const float sharp = ccs_CreaseSharpness(subd, edgeID, depth);
        const float edgeWeight = cc__Satf(sharp);
        const cc_VertexPoint oldEdgePoints[2] = {
            ccs_HalfedgeVertexPoint(subd, halfedgeID, depth),
            ccs_HalfedgeVertexPoint(subd,     nextID, depth)
        };
        const cc_VertexPoint newAdjacentFacePoints[2] = {
            newFacePoints[ccs_HalfedgeFaceID(subd,         halfedgeID, depth)],
            newFacePoints[ccs_HalfedgeFaceID(subd, cc__Max(0, twinID), depth)]
        };
        cc_VertexPoint sharpEdgePoint = {0.0f, 0.0f, 0.0f};
        cc_VertexPoint smoothEdgePoint = {0.0f, 0.0f, 0.0f};
        float tmp1[3], tmp2[3];

        cc__Add3f(tmp1, oldEdgePoints[0].array, oldEdgePoints[1].array);
        cc__Add3f(tmp2, newAdjacentFacePoints[0].array, newAdjacentFacePoints[1].array);
        cc__Mul3f(sharpEdgePoint.array, tmp1, 0.5f);
        cc__Add3f(smoothEdgePoint.array, tmp1, tmp2);
        cc__Mul3f(smoothEdgePoint.array, smoothEdgePoint.array, 0.25f);
        cc__Lerp3f(newEdgePoints[edgeID].array,
                   smoothEdgePoint.array,
                   sharpEdgePoint.array,
                   edgeWeight);
    }
    __syncthreads();
}


__global__ void ccs__CreasedEdgePoints_Scatter(cc_Subd *subd, int32_t depth)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t vertexCount = ccm_VertexCountAtDepth_Fast(cage, depth);
    const int32_t faceCount = ccm_FaceCountAtDepth_Fast(cage, depth);
    const int32_t halfedgeCount = ccm_HalfedgeCountAtDepth(cage, depth);
    const int32_t stride = ccs_CumulativeVertexCountAtDepth(cage, depth);
    const cc_VertexPoint *newFacePoints = &subd->vertexPoints[stride + vertexCount];
    cc_VertexPoint *newEdgePoints = &subd->vertexPoints[stride + vertexCount + faceCount];

    int edges_per_thread = std::ceil(float(halfedgeCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t halfedgeID = start; halfedgeID < halfedgeCount && halfedgeID < end; ++halfedgeID) {
        const int32_t twinID = ccs_HalfedgeTwinID(subd, halfedgeID, depth);
        const int32_t edgeID = ccs_HalfedgeEdgeID(subd, halfedgeID, depth);
        const int32_t faceID = ccs_HalfedgeFaceID(subd, halfedgeID, depth);
        const int32_t nextID = ccs_HalfedgeNextID(subd, halfedgeID, depth);
        const float sharp = ccs_CreaseSharpness(subd, edgeID, depth);
        const float edgeWeight = cc__Satf(sharp);
        const cc_VertexPoint newFacePoint = newFacePoints[faceID];
        const cc_VertexPoint oldEdgePoints[2] = {
            ccs_HalfedgeVertexPoint(subd, halfedgeID, depth),
            ccs_HalfedgeVertexPoint(subd,     nextID, depth)
        };
        cc_VertexPoint smoothPoint = {0.0f, 0.0f, 0.0f};
        cc_VertexPoint sharpPoint = {0.0f, 0.0f, 0.0f};
        float tmp[3], atomicWeight[3];

        // sharp point
        cc__Lerp3f(tmp, oldEdgePoints[0].array, oldEdgePoints[1].array, 0.5f);
        cc__Mul3f(sharpPoint.array, tmp, twinID < 0 ? 1.0f : 0.5f);

        // smooth point
        cc__Lerp3f(tmp, oldEdgePoints[0].array, newFacePoint.array, 0.5f);
        cc__Mul3f(smoothPoint.array, tmp, 0.5f);

        // atomic weight
        cc__Lerp3f(atomicWeight,
                   smoothPoint.array,
                   sharpPoint.array,
                   edgeWeight);

        for (int32_t i = 0; i < 3; ++i) {
// #pragma omp atomic
            // newEdgePoints[edgeID].array[i]+= atomicWeight[i];
            atomicAdd(newEdgePoints[edgeID].array + i, atomicWeight[i]);
        }
    }
    __syncthreads();
}


/*******************************************************************************
 * VertexPoints -- Applies Catmull Clark's vertex rule on the subd
 *
 * The "Gather" routine iterates over each vertex of the mesh and computes the
 * resulting smooth vertex.
 *
 * The "Scatter" routine iterates over each halfedge of the mesh and atomically
 * adds its contribution to the computation of the smooth vertex.
 *
 */
__global__ void ccs__VertexPoints_Gather(cc_Subd *subd, int32_t depth)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t vertexCount = ccm_VertexCountAtDepth_Fast(cage, depth);
    const int32_t faceCount = ccm_FaceCountAtDepth_Fast(cage, depth);
    const int32_t stride = ccs_CumulativeVertexCountAtDepth(cage, depth);
    const cc_VertexPoint *newFacePoints = &subd->vertexPoints[stride + vertexCount];
    const cc_VertexPoint *newEdgePoints = &subd->vertexPoints[stride + vertexCount + faceCount];
    cc_VertexPoint *newVertexPoints = &subd->vertexPoints[stride];
    
    int edges_per_thread = std::ceil(float(vertexCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t vertexID = start; vertexID < vertexCount && vertexID < end; ++vertexID) {
        const int32_t halfedgeID = ccs_VertexPointToHalfedgeID(subd, vertexID, depth);
        const int32_t edgeID = ccs_HalfedgeEdgeID(subd, halfedgeID, depth);
        const int32_t faceID = ccs_HalfedgeFaceID(subd, halfedgeID, depth);
        const cc_VertexPoint newEdgePoint = newEdgePoints[edgeID];
        const cc_VertexPoint newFacePoint = newFacePoints[faceID];
        const cc_VertexPoint oldVertexPoint = ccs_VertexPoint(subd, vertexID, depth);
        cc_VertexPoint smoothPoint = {0.0f, 0.0f, 0.0f};
        float valence = 1.0f;
        int32_t iterator;
        float tmp1[3], tmp2[3];

        cc__Mul3f(tmp1, newFacePoint.array, -1.0f);
        cc__Mul3f(tmp2, newEdgePoint.array, +4.0f);
        cc__Add3f(smoothPoint.array, tmp1, tmp2);

        for (iterator = ccs_PrevVertexHalfedgeID(subd, halfedgeID, depth);
             iterator >= 0 && iterator != halfedgeID;
             iterator = ccs_PrevVertexHalfedgeID(subd, iterator, depth)) {
            const int32_t edgeID = ccs_HalfedgeEdgeID(subd, iterator, depth);
            const int32_t faceID = ccs_HalfedgeFaceID(subd, iterator, depth);
            const cc_VertexPoint newEdgePoint = newEdgePoints[edgeID];
            const cc_VertexPoint newFacePoint = newFacePoints[faceID];

            cc__Mul3f(tmp1, newFacePoint.array, -1.0f);
            cc__Mul3f(tmp2, newEdgePoint.array, +4.0f);
            cc__Add3f(smoothPoint.array, smoothPoint.array, tmp1);
            cc__Add3f(smoothPoint.array, smoothPoint.array, tmp2);
            ++valence;
        }

        cc__Mul3f(tmp1, smoothPoint.array, 1.0f / (valence * valence));
        cc__Mul3f(tmp2, oldVertexPoint.array, 1.0f - 3.0f / valence);
        cc__Add3f(smoothPoint.array, tmp1, tmp2);
        cc__Lerp3f(newVertexPoints[vertexID].array,
                   oldVertexPoint.array,
                   smoothPoint.array,
                   iterator != halfedgeID ? 0.0f : 1.0f);
    }
    __syncthreads();
}

__global__ void ccs__VertexPoints_Scatter(cc_Subd *subd, int32_t depth)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t vertexCount = ccm_VertexCountAtDepth_Fast(cage, depth);
    const int32_t faceCount = ccm_FaceCountAtDepth_Fast(cage, depth);
    const int32_t halfedgeCount = ccm_HalfedgeCountAtDepth(cage, depth);
    const int32_t stride = ccs_CumulativeVertexCountAtDepth(cage, depth);
    const cc_VertexPoint *newFacePoints = &subd->vertexPoints[stride + vertexCount];
    const cc_VertexPoint *newEdgePoints = &subd->vertexPoints[stride + vertexCount + faceCount];
    cc_VertexPoint *newVertexPoints = &subd->vertexPoints[stride];

    int edges_per_thread = std::ceil(float(halfedgeCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t halfedgeID = start; halfedgeID < halfedgeCount && halfedgeID < end; ++halfedgeID) {
        const int32_t vertexID = ccs_HalfedgeVertexID(subd, halfedgeID, depth);
        const int32_t edgeID = ccs_HalfedgeEdgeID(subd, halfedgeID, depth);
        const int32_t faceID = ccs_HalfedgeFaceID(subd, halfedgeID, depth);
        const cc_VertexPoint oldVertexPoint = ccs_VertexPoint(subd, vertexID, depth);
        int32_t valence = 1;
        int32_t forwardIterator, backwardIterator;

        for (forwardIterator = ccs_PrevVertexHalfedgeID(subd, halfedgeID, depth);
             forwardIterator >= 0 && forwardIterator != halfedgeID;
             forwardIterator = ccs_PrevVertexHalfedgeID(subd, forwardIterator, depth)) {
            ++valence;
        }

        for (backwardIterator = ccs_NextVertexHalfedgeID(subd, halfedgeID, depth);
             forwardIterator < 0 && backwardIterator >= 0 && backwardIterator != halfedgeID;
             backwardIterator = ccs_NextVertexHalfedgeID(subd, backwardIterator, depth)) {
            ++valence;
        }

        for (int32_t i = 0; i < 3; ++i) {
            const float w = 1.0f / (float)valence;
            const float v = oldVertexPoint.array[i];
            const float f = newFacePoints[faceID].array[i];
            const float e = newEdgePoints[edgeID].array[i];
            const float s = forwardIterator < 0 ? 0.0f : 1.0f;
// #pragma omp atomic
            // newVertexPoints[vertexID].array[i]+=
                // w * (v + w * s * (4.0f * e - f - 3.0f * v));
            atomicAdd(newVertexPoints[vertexID].array + i, w * (v + w * s * (4.0f * e - f - 3.0f * v)));
        }
    }
    __syncthreads();
}


/*******************************************************************************
 * CreasedVertexPoints -- Applies DeRose et al.'s vertex rule on the subd
 *
 * The "Gather" routine iterates over each vertex of the mesh and computes the
 * resulting smooth vertex.
 *
 * The "Scatter" routine iterates over each halfedge of the mesh and atomically
 * adds its contribution to the computation of the smooth vertex.
 *
 */
__global__ void ccs__CreasedVertexPoints_Gather(cc_Subd *subd, int32_t depth)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t vertexCount = ccm_VertexCountAtDepth_Fast(cage, depth);
    const int32_t faceCount = ccm_FaceCountAtDepth_Fast(cage, depth);
    const int32_t stride = ccs_CumulativeVertexCountAtDepth(cage, depth);
    const cc_VertexPoint *newFacePoints = &subd->vertexPoints[stride + vertexCount];
    const cc_VertexPoint *newEdgePoints = &subd->vertexPoints[stride + vertexCount + faceCount];
    cc_VertexPoint *newVertexPoints = &subd->vertexPoints[stride];

    int edges_per_thread = std::ceil(float(vertexCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t vertexID = start; vertexID < end && vertexID < vertexCount; ++vertexID) {
        const int32_t halfedgeID = ccs_VertexPointToHalfedgeID(subd, vertexID, depth);
        const int32_t edgeID = ccs_HalfedgeEdgeID(subd, halfedgeID, depth);
        const int32_t prevID = ccs_HalfedgePrevID(subd, halfedgeID, depth);
        const int32_t prevEdgeID = ccs_HalfedgeEdgeID(subd, prevID, depth);
        const int32_t prevFaceID = ccs_HalfedgeFaceID(subd, prevID, depth);
        const float thisS = ccs_HalfedgeSharpness(subd, halfedgeID, depth);
        const float prevS = ccs_HalfedgeSharpness(subd,     prevID, depth);
        const float creaseWeight = cc__Signf(thisS);
        const float prevCreaseWeight = cc__Signf(prevS);
        const cc_VertexPoint newEdgePoint = newEdgePoints[edgeID];
        const cc_VertexPoint newPrevEdgePoint = newEdgePoints[prevEdgeID];
        const cc_VertexPoint newPrevFacePoint = newFacePoints[prevFaceID];
        const cc_VertexPoint oldPoint = ccs_VertexPoint(subd, vertexID, depth);
        cc_VertexPoint smoothPoint = {0.0f, 0.0f, 0.0f};
        cc_VertexPoint creasePoint = {0.0f, 0.0f, 0.0f};
        float avgS = prevS;
        float creaseCount = prevCreaseWeight;
        float valence = 1.0f;
        int32_t forwardIterator, backwardIterator;
        float tmp1[3], tmp2[3];

        // smooth contrib
        cc__Mul3f(tmp1, newPrevFacePoint.array, -1.0f);
        cc__Mul3f(tmp2, newPrevEdgePoint.array, +4.0f);
        cc__Add3f(smoothPoint.array, tmp1, tmp2);

        // crease contrib
        cc__Mul3f(tmp1, newPrevEdgePoint.array, prevCreaseWeight);
        cc__Add3f(creasePoint.array, creasePoint.array, tmp1);

        for (forwardIterator = ccs_HalfedgeTwinID(subd, prevID, depth);
             forwardIterator >= 0 && forwardIterator != halfedgeID;
             forwardIterator = ccs_HalfedgeTwinID(subd, forwardIterator, depth)) {
            const int32_t prevID = ccs_HalfedgePrevID(subd, forwardIterator, depth);
            const int32_t prevEdgeID = ccs_HalfedgeEdgeID(subd, prevID, depth);
            const int32_t prevFaceID = ccs_HalfedgeFaceID(subd, prevID, depth);
            const cc_VertexPoint newPrevEdgePoint = newEdgePoints[prevEdgeID];
            const cc_VertexPoint newPrevFacePoint = newFacePoints[prevFaceID];
            const float prevS = ccs_HalfedgeSharpness(subd, prevID, depth);
            const float prevCreaseWeight = cc__Signf(prevS);

            // smooth contrib
            cc__Mul3f(tmp1, newPrevFacePoint.array, -1.0f);
            cc__Mul3f(tmp2, newPrevEdgePoint.array, +4.0f);
            cc__Add3f(smoothPoint.array, smoothPoint.array, tmp1);
            cc__Add3f(smoothPoint.array, smoothPoint.array, tmp2);
            ++valence;

            // crease contrib
            cc__Mul3f(tmp1, newPrevEdgePoint.array, prevCreaseWeight);
            cc__Add3f(creasePoint.array, creasePoint.array, tmp1);
            avgS+= prevS;
            creaseCount+= prevCreaseWeight;

            // next vertex halfedge
            forwardIterator = prevID;
        }

        for (backwardIterator = ccs_HalfedgeTwinID(subd, halfedgeID, depth);
             forwardIterator < 0 && backwardIterator >= 0 && backwardIterator != halfedgeID;
             backwardIterator = ccs_HalfedgeTwinID(subd, backwardIterator, depth)) {
            const int32_t nextID = ccs_HalfedgeNextID(subd, backwardIterator, depth);
            const int32_t nextEdgeID = ccs_HalfedgeEdgeID(subd, nextID, depth);
            const int32_t nextFaceID = ccs_HalfedgeFaceID(subd, nextID, depth);
            const cc_VertexPoint newNextEdgePoint = newEdgePoints[nextEdgeID];
            const cc_VertexPoint newNextFacePoint = newFacePoints[nextFaceID];
            const float nextS = ccs_HalfedgeSharpness(subd, nextID, depth);
            const float nextCreaseWeight = cc__Signf(nextS);

            // smooth contrib
            cc__Mul3f(tmp1, newNextFacePoint.array, -1.0f);
            cc__Mul3f(tmp2, newNextEdgePoint.array, +4.0f);
            cc__Add3f(smoothPoint.array, smoothPoint.array, tmp1);
            cc__Add3f(smoothPoint.array, smoothPoint.array, tmp2);
            ++valence;

            // crease contrib
            cc__Mul3f(tmp1, newNextEdgePoint.array, nextCreaseWeight);
            cc__Add3f(creasePoint.array, creasePoint.array, tmp1);
            avgS+= nextS;
            creaseCount+= nextCreaseWeight;

            // next vertex halfedge
            backwardIterator = nextID;
        }

        // boundary corrections
        if (forwardIterator < 0) {
            cc__Mul3f(tmp1, newEdgePoint.array    , creaseWeight);
            cc__Add3f(creasePoint.array, creasePoint.array, tmp1);
            creaseCount+= creaseWeight;
            ++valence;
        }

        // smooth point
        cc__Mul3f(tmp1, smoothPoint.array, 1.0f / (valence * valence));
        cc__Mul3f(tmp2, oldPoint.array, 1.0f - 3.0f / valence);
        cc__Add3f(smoothPoint.array, tmp1, tmp2);

        // crease point
        cc__Mul3f(tmp1, creasePoint.array, 0.5f / creaseCount);
        cc__Mul3f(tmp2, oldPoint.array, 0.5f);
        cc__Add3f(creasePoint.array, tmp1, tmp2);

        // proper vertex rule selection (TODO: make branchless)
        if (creaseCount <= 1.0f) {
            newVertexPoints[vertexID] = smoothPoint;
        } else if (creaseCount >= 3.0f || valence == 2.0f) {
            newVertexPoints[vertexID] = oldPoint;
        } else {
            cc__Lerp3f(newVertexPoints[vertexID].array,
                       oldPoint.array,
                       creasePoint.array,
                       cc__Satf(avgS * 0.5f));
        }
    }
    __syncthreads();
}


__global__ void ccs__CreasedVertexPoints_Scatter(cc_Subd *subd, int32_t depth)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t halfedgeCount = ccm_HalfedgeCountAtDepth(cage, depth);
    const int32_t vertexCount = ccm_VertexCountAtDepth_Fast(cage, depth);
    const int32_t faceCount = ccm_FaceCountAtDepth_Fast(cage, depth);
    const int32_t stride = ccs_CumulativeVertexCountAtDepth(cage, depth);
    const cc_VertexPoint *newFacePoints = &subd->vertexPoints[stride + vertexCount];
    const cc_VertexPoint *newEdgePoints = &subd->vertexPoints[stride + vertexCount + faceCount];
    cc_VertexPoint *newVertexPoints = &subd->vertexPoints[stride];

    int edges_per_thread = std::ceil(float(halfedgeCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    printf("halfedgeCount %d\n", halfedgeCount);
    for (int32_t halfedgeID = start; halfedgeID < end && halfedgeID < halfedgeCount; ++halfedgeID) {
        const int32_t vertexID = ccs_HalfedgeVertexID(subd, halfedgeID, depth);
        const int32_t edgeID = ccs_HalfedgeEdgeID(subd, halfedgeID, depth);
        const int32_t faceID = ccs_HalfedgeFaceID(subd, halfedgeID, depth);
        const int32_t prevID = ccs_HalfedgePrevID(subd, halfedgeID, depth);
        const int32_t prevEdgeID = ccs_HalfedgeEdgeID(subd, prevID, depth);
        const float thisS = ccs_HalfedgeSharpness(subd, halfedgeID, depth);
        const float prevS = ccs_HalfedgeSharpness(subd,     prevID, depth);
        const float creaseWeight = cc__Signf(thisS);
        const float prevCreaseWeight = cc__Signf(prevS);
        const cc_VertexPoint newPrevEdgePoint = newEdgePoints[prevEdgeID];
        const cc_VertexPoint newEdgePoint = newEdgePoints[edgeID];
        const cc_VertexPoint newFacePoint = newFacePoints[faceID];
        const cc_VertexPoint oldPoint = ccs_VertexPoint(subd, vertexID, depth);
        cc_VertexPoint cornerPoint = {0.0f, 0.0f, 0.0f};
        cc_VertexPoint smoothPoint = {0.0f, 0.0f, 0.0f};
        cc_VertexPoint creasePoint = {0.0f, 0.0f, 0.0f};
        cc_VertexPoint atomicWeight = {0.0f, 0.0f, 0.0f};
        float avgS = prevS;
        float creaseCount = prevCreaseWeight;
        float valence = 1.0f;
        int32_t forwardIterator, backwardIterator;
        float tmp1[3], tmp2[3];
        // printf("Before the forward loop %d\n", start);
        bool isHalfedgeZero = 0;
        if(halfedgeID == 0){
            isHalfedgeZero = 1;
        }
        for (forwardIterator = ccs_HalfedgeTwinID(subd, prevID, depth);
             forwardIterator >= 0 && forwardIterator != halfedgeID;
             forwardIterator = ccs_HalfedgeTwinID(subd, forwardIterator, depth)) {
            const int32_t prevID = ccs_HalfedgePrevID(subd, forwardIterator, depth);
            const float prevS = ccs_HalfedgeSharpness(subd, prevID, depth);
            const float prevCreaseWeight = cc__Signf(prevS);

            // valence computation
            ++valence;

            // crease computation
            avgS+= prevS;
            creaseCount+= prevCreaseWeight;

            // next vertex halfedge
            forwardIterator = prevID;
            if(isHalfedgeZero){
                printf("Forward %d == %d ? \n", forwardIterator, halfedgeID);
                int32_t twin = ccs_HalfedgeTwinID(subd, forwardIterator, depth);
                printf("Twin %d ? \n", twin);
            }
        }
        printf("Before the backward loop %d\n", start);
        for (backwardIterator = ccs_HalfedgeTwinID(subd, halfedgeID, depth);
             forwardIterator < 0 && backwardIterator >= 0 && backwardIterator != halfedgeID;
             backwardIterator = ccs_HalfedgeTwinID(subd, backwardIterator, depth)) {
            const int32_t nextID = ccs_HalfedgeNextID(subd, backwardIterator, depth);
            const float nextS = ccs_HalfedgeSharpness(subd, nextID, depth);
            const float nextCreaseWeight = cc__Signf(nextS);

            // valence computation
            ++valence;

            // crease computation
            avgS+= nextS;
            creaseCount+= nextCreaseWeight;

            // next vertex halfedge
            backwardIterator = nextID;
        }

        // corner point
        cc__Mul3f(cornerPoint.array, oldPoint.array, 1.0f / valence);

        // crease computation: V / 4
        cc__Mul3f(tmp1, oldPoint.array, 0.25f * creaseWeight);
        cc__Mul3f(tmp2, newEdgePoint.array, 0.25f * creaseWeight);
        cc__Add3f(creasePoint.array, tmp1, tmp2);

        // smooth computation: (4E - F + (n - 3) V) / N
        cc__Mul3f(tmp1, newFacePoint.array, -1.0f);
        cc__Mul3f(tmp2, newEdgePoint.array, +4.0f);
        cc__Add3f(smoothPoint.array, tmp1, tmp2);
        cc__Mul3f(tmp1, oldPoint.array, valence - 3.0f);
        cc__Add3f(smoothPoint.array, smoothPoint.array, tmp1);
        cc__Mul3f(smoothPoint.array,
                  smoothPoint.array,
                  1.0f / (valence * valence));

        // boundary corrections
        if (forwardIterator < 0) {
            creaseCount+= creaseWeight;
            ++valence;

            cc__Mul3f(tmp1, oldPoint.array, 0.25f * prevCreaseWeight);
            cc__Mul3f(tmp2, newPrevEdgePoint.array, 0.25f * prevCreaseWeight);
            cc__Add3f(tmp1, tmp1, tmp2);
            cc__Add3f(creasePoint.array, creasePoint.array, tmp1);
        }

        // atomicWeight (TODO: make branchless ?)
        if (creaseCount >= 3.0f || valence == 2.0f) {
            atomicWeight = cornerPoint;
        } else if (creaseCount <= 1.0f) {
            atomicWeight = smoothPoint;
        } else {
            cc__Lerp3f(atomicWeight.array,
                       cornerPoint.array,
                       creasePoint.array,
                       cc__Satf(avgS * 0.5f));
        }

        for (int32_t i = 0; i < 3; ++i) {
// #pragma omp atomic
            // newVertexPoints[vertexID].array[i]+= atomicWeight.array[i];
            atomicAdd(newVertexPoints[vertexID].array + i, atomicWeight.array[i]);
        }
    }
    __syncthreads();
    printf("After Sync Threads %d\n", start);
}



/*******************************************************************************
 * RefineVertexPoints -- Computes the result of Catmull Clark subdivision.
 *
 */
void ccs__ClearVertexPoints(cc_Subd *subd)
{
    const int32_t vertexCount = ccs_CumulativeVertexCount(subd);
    const int32_t vertexByteCount = vertexCount * sizeof(cc_VertexPoint);

    CC_MEMSET(subd->vertexPoints, 0, vertexByteCount);
}

 void ccs_RefineVertexPoints_Scatter(cc_Subd *subd)
{
    ccs__ClearVertexPoints(subd);
    ccs__CageFacePoints_Scatter<<<1, NUM_THREADS>>>(subd);
    ccs__CreasedCageEdgePoints_Scatter<<<1, NUM_THREADS>>>(subd);
    ccs__CreasedCageVertexPoints_Scatter<<<1, NUM_THREADS>>>(subd);

    for (int32_t depth = 1; depth < ccs_MaxDepth(subd); ++depth) {
        ccs__FacePoints_Scatter<<<1, NUM_THREADS>>>(subd, depth);
        ccs__CreasedEdgePoints_Scatter<<<1, NUM_THREADS>>>(subd, depth);
        ccs__CreasedVertexPoints_Scatter<<<1, NUM_THREADS>>>(subd, depth);
        cudaDeviceSynchronize();
    }
}

 void ccs_RefineVertexPoints_NoCreases_Scatter(cc_Subd *subd)
{
    ccs__ClearVertexPoints(subd);
    ccs__CageFacePoints_Scatter<<<1, NUM_THREADS>>>(subd);
    ccs__CageEdgePoints_Scatter<<<1, NUM_THREADS>>>(subd);
    ccs__CageVertexPoints_Scatter<<<1, NUM_THREADS>>>(subd);

    for (int32_t depth = 1; depth < ccs_MaxDepth(subd); ++depth) {
        ccs__FacePoints_Scatter<<<1, NUM_THREADS>>>(subd, depth);
        ccs__EdgePoints_Scatter<<<1, NUM_THREADS>>>(subd, depth);
        ccs__VertexPoints_Scatter<<<1, NUM_THREADS>>>(subd, depth);
    }
}

 void ccs_RefineVertexPoints_Gather(cc_Subd *subd)
{
    ccs__CageFacePoints_Gather<<<1, NUM_THREADS>>>(subd);
    ccs__CreasedCageEdgePoints_Gather<<<1, NUM_THREADS>>>(subd);
    ccs__CreasedCageVertexPoints_Gather<<<1, NUM_THREADS>>>(subd);

    for (int32_t depth = 1; depth < ccs_MaxDepth(subd); ++depth) {
        ccs__FacePoints_Gather<<<1, NUM_THREADS>>>(subd, depth);
        ccs__CreasedEdgePoints_Gather<<<1, NUM_THREADS>>>(subd, depth);
        ccs__CreasedVertexPoints_Gather<<<1, NUM_THREADS>>>(subd, depth);
    }
}

 void ccs_RefineVertexPoints_NoCreases_Gather(cc_Subd *subd)
{
    ccs__CageFacePoints_Gather<<<1, NUM_THREADS>>>(subd);
    ccs__CageEdgePoints_Gather<<<1, NUM_THREADS>>>(subd);
    ccs__CageVertexPoints_Gather<<<1, NUM_THREADS>>>(subd);

    for (int32_t depth = 1; depth < ccs_MaxDepth(subd); ++depth) {
        ccs__FacePoints_Gather<<<1, NUM_THREADS>>>(subd, depth);
        ccs__EdgePoints_Gather<<<1, NUM_THREADS>>>(subd, depth);
        ccs__VertexPoints_Gather<<<1, NUM_THREADS>>>(subd, depth);
    }
}


/*******************************************************************************
 * RefineCageHalfedges -- Applies halfedge refinement rules on the cage mesh
 *
 * This routine computes the halfedges of the control cage after one subdivision
 * step and stores them in the subd.
 *
 */
// __global__ void ccs__RefineCageHalfedges(const cc_Mesh *cage, int32_t vertexCount, int32_t edgeCount, int32_t faceCount, int32_t halfedgeCount, cc_Halfedge_SemiRegular *halfedgesOut)
// {
//     int halfedgeID = threadIdx.x;
//     if (halfedgeID >= halfedgeCount) return;
//     // for (int32_t halfedgeID = start; halfedgeID < end && halfedgeID < halfedgeCount; ++halfedgeID) {
//     const int32_t twinID = ccm_HalfedgeTwinID(cage, halfedgeID);
//     const int32_t prevID = ccm_HalfedgePrevID(cage, halfedgeID);
//     const int32_t nextID = ccm_HalfedgeNextID(cage, halfedgeID);
//     const int32_t faceID = ccm_HalfedgeFaceID(cage, halfedgeID);
//     const int32_t edgeID = ccm_HalfedgeEdgeID(cage, halfedgeID);
//     const int32_t prevEdgeID = ccm_HalfedgeEdgeID(cage, prevID);
//     const int32_t prevTwinID = ccm_HalfedgeTwinID(cage, prevID);
//     const int32_t vertexID = ccm_HalfedgeVertexID(cage, halfedgeID);
//     const int32_t twinNextID =
//         twinID >= 0 ? ccm_HalfedgeNextID(cage, twinID) : -1;
//     cc_Halfedge_SemiRegular *newHalfedges[4] = {
//         &halfedgesOut[(4 * halfedgeID + 0)],
//         &halfedgesOut[(4 * halfedgeID + 1)],
//         &halfedgesOut[(4 * halfedgeID + 2)],
//         &halfedgesOut[(4 * halfedgeID + 3)]
//     };

//     // twinIDs
//     newHalfedges[0]->twinID = 4 * twinNextID + 3;
//     newHalfedges[1]->twinID = 4 * nextID     + 2;
//     newHalfedges[2]->twinID = 4 * prevID     + 1;
//     newHalfedges[3]->twinID = 4 * prevTwinID + 0;

//     // edgeIDs
//     newHalfedges[0]->edgeID = 2 * edgeID + (halfedgeID > twinID ? 0 : 1);
//     newHalfedges[1]->edgeID = 2 * edgeCount + halfedgeID;
//     newHalfedges[2]->edgeID = 2 * edgeCount + prevID;
//     newHalfedges[3]->edgeID = 2 * prevEdgeID + (prevID > prevTwinID ? 1 : 0);

//     // vertexIDs
//     newHalfedges[0]->vertexID = vertexID;
//     newHalfedges[1]->vertexID = vertexCount + faceCount + edgeID;
//     newHalfedges[2]->vertexID = vertexCount + faceID;
//     newHalfedges[3]->vertexID = vertexCount + faceCount + prevEdgeID;
//     //     if(halfedgeID == 0){ // seems to be correct here. 
//     //         printf("twinID %d, edgeID %d, vertexID %d \n", halfedgesOut[(4 * halfedgeID + 1)].twinID, halfedgesOut[(4 * halfedgeID + 1)].edgeID, halfedgesOut[(4 * halfedgeID + 1)].vertexID);
//     //     }
//     // }
//     // __syncthreads();
// }


// /*******************************************************************************
//  * RefineHalfedges -- Applies halfedge refinement on the subd
//  *
//  * This routine computes the halfedges of the next subd level.
//  *
//  */
// //  void ccs__RefineHalfedges(cc_Subd *subd, int32_t depth)
// // {
// //     const cc_Mesh *cage = subd->cage;
// //     const int32_t halfedgeCount = ccm_HalfedgeCountAtDepth(cage, depth);
// //     const int32_t vertexCount = ccm_VertexCountAtDepth_Fast(cage, depth);
// //     const int32_t edgeCount = ccm_EdgeCountAtDepth_Fast(cage, depth);
// //     const int32_t faceCount = ccm_FaceCountAtDepth_Fast(cage, depth);
// //     const int32_t stride = ccs_CumulativeHalfedgeCountAtDepth(cage, depth);
// //     cc_Halfedge_SemiRegular *halfedgesOut = &subd->halfedges[stride];
// //     rewrite_RefineHalfedges<<<1, NUMTHREADS>>>(cage, halfedgeCount, vertexCount, edgeCount, faceCount, stride, halfedgesOut);
// // }

// __global__ void rewrite_RefineHalfedges(cc_Subd *subd, int32_t depth, const cc_Mesh *cage, int32_t halfedgeCount, int32_t vertexCount, int32_t edgeCount, int32_t faceCount, int32_t stride, cc_Halfedge_SemiRegular *halfedgesOut)
// {
//     int halfedgeID = threadIdx.x;
//     if (halfedgeID >= halfedgeCount) return;
//     // int edges_per_thread = std::ceil(float(halfedgeCount) / float(NUM_THREADS));
//     // int start = threadIdx.x;
//     // int end = threadIdx.x + edges_per_thread;
//     // for (int32_t halfedgeID = start; halfedgeID < end && halfedgeID < halfedgeCount; ++halfedgeID) {
//         const int32_t twinID = ccs_HalfedgeTwinID(subd, halfedgeID, depth);
//         const int32_t prevID = ccm_HalfedgePrevID_Quad(halfedgeID);
//         const int32_t nextID = ccm_HalfedgeNextID_Quad(halfedgeID);
//         const int32_t faceID = ccm_HalfedgeFaceID_Quad(halfedgeID);
//         const int32_t edgeID = ccs_HalfedgeEdgeID(subd, halfedgeID, depth);
//         const int32_t vertexID = ccs_HalfedgeVertexID(subd, halfedgeID, depth);
//         const int32_t prevEdgeID = ccs_HalfedgeEdgeID(subd, prevID, depth);
//         const int32_t prevTwinID = ccs_HalfedgeTwinID(subd, prevID, depth);
//         const int32_t twinNextID = ccm_HalfedgeNextID_Quad(twinID);
//         cc_Halfedge_SemiRegular *newHalfedges[4] = {
//             &halfedgesOut[(4 * halfedgeID + 0)],
//             &halfedgesOut[(4 * halfedgeID + 1)],
//             &halfedgesOut[(4 * halfedgeID + 2)],
//             &halfedgesOut[(4 * halfedgeID + 3)]
//         };

//         // twinIDs
//         newHalfedges[0]->twinID = 4 * twinNextID + 3;
//         newHalfedges[1]->twinID = 4 * nextID     + 2;
//         newHalfedges[2]->twinID = 4 * prevID     + 1;
//         newHalfedges[3]->twinID = 4 * prevTwinID + 0;

//         // edgeIDs
//         newHalfedges[0]->edgeID = 2 * edgeID + (halfedgeID > twinID ? 0 : 1);
//         newHalfedges[1]->edgeID = 2 * edgeCount + halfedgeID;
//         newHalfedges[2]->edgeID = 2 * edgeCount + prevID;
//         newHalfedges[3]->edgeID = 2 * prevEdgeID + (prevID > prevTwinID ? 1 : 0);

//         // vertexIDs
//         newHalfedges[0]->vertexID = vertexID;
//         newHalfedges[1]->vertexID = vertexCount + faceCount + edgeID;
//         newHalfedges[2]->vertexID = vertexCount + faceID;
//         newHalfedges[3]->vertexID = vertexCount + faceCount + prevEdgeID;
//         // if(halfedgeID == 0){ // seems to be correct here. 
//         //     printf("twinID %d, edgeID %d, vertexID %d \n", halfedgesOut[(4 * halfedgeID + 1)].twinID, halfedgesOut[(4 * halfedgeID + 1)].edgeID, halfedgesOut[(4 * halfedgeID + 1)].vertexID);
//         // }
//     // }
// }


// /*******************************************************************************
//  * RefineHalfedges
//  *
//  */
//  void ccs_RefineHalfedges(cc_Subd *subd)
// {   
//     const int32_t maxDepth = ccs_MaxDepth(subd);

//     const cc_Mesh *cage = subd->cage;
//     int32_t vertexCount = ccm_VertexCount(cage);
//     int32_t edgeCount = ccm_EdgeCount(cage);
//     int32_t faceCount = ccm_FaceCount(cage);
//     int32_t halfedgeCount = ccm_HalfedgeCount(cage);
//     cc_Halfedge_SemiRegular *halfedgesOut = subd->halfedges;

//     // ccs__RefineCageHalfedges<<<1, NUM_THREADS>>>(subd);
//     ccs__RefineCageHalfedges<<<(halfedgeCount + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(cage, vertexCount, edgeCount, faceCount, halfedgeCount, halfedgesOut);
//     for (int32_t depth = 1; depth < maxDepth; ++depth) {
//         int32_t halfedgeCount1 = ccm_HalfedgeCountAtDepth(cage, depth);
//         int32_t vertexCount1 = ccm_VertexCountAtDepth_Fast(cage, depth);
//         int32_t edgeCount1 = ccm_EdgeCountAtDepth_Fast(cage, depth);
//         int32_t faceCount1 = ccm_FaceCountAtDepth_Fast(cage, depth);
//         int32_t stride1 = ccs_CumulativeHalfedgeCountAtDepth(cage, depth);
//         cc_Halfedge_SemiRegular *halfedgesOut1 = &subd->halfedges[stride1];
//         rewrite_RefineHalfedges<<<(halfedgeCount + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(subd, depth, cage, halfedgeCount1, vertexCount1, edgeCount1, faceCount1, stride1, halfedgesOut1);
//         cudaDeviceSynchronize();
//     }
// }

/*******************************************************************************
 * RefineCageHalfedges -- Applies halfedge refinement rules on the cage mesh
 *
 * This routine computes the halfedges of the control cage after one subdivision
 * step and stores them in the subd.
 *
 */
__global__ void RefineCageInner(const cc_Mesh *cage, int32_t vertexCount, int32_t edgeCount, int32_t faceCount, int32_t halfedgeCount, cc_Halfedge_SemiRegular *halfedgesOut){
    CHECK_TID(halfedgeCount)
    int32_t halfedgeID = TID;
    const int32_t twinID = ccm_HalfedgeTwinID(cage, halfedgeID);
    const int32_t prevID = ccm_HalfedgePrevID(cage, halfedgeID);
    const int32_t nextID = ccm_HalfedgeNextID(cage, halfedgeID);
    const int32_t faceID = ccm_HalfedgeFaceID(cage, halfedgeID);
    const int32_t edgeID = ccm_HalfedgeEdgeID(cage, halfedgeID);
    const int32_t prevEdgeID = ccm_HalfedgeEdgeID(cage, prevID);
    const int32_t prevTwinID = ccm_HalfedgeTwinID(cage, prevID);
    const int32_t vertexID = ccm_HalfedgeVertexID(cage, halfedgeID);
    const int32_t twinNextID = twinID >= 0 ? ccm_HalfedgeNextID(cage, twinID) : -1;
    
    cc_Halfedge_SemiRegular *newHalfedges[4] = {
        &halfedgesOut[(4 * halfedgeID + 0)],
        &halfedgesOut[(4 * halfedgeID + 1)],
        &halfedgesOut[(4 * halfedgeID + 2)],
        &halfedgesOut[(4 * halfedgeID + 3)]
    };

    // twinIDs
    newHalfedges[0]->twinID = 4 * twinNextID + 3;
    newHalfedges[1]->twinID = 4 * nextID     + 2;
    newHalfedges[2]->twinID = 4 * prevID     + 1;
    newHalfedges[3]->twinID = 4 * prevTwinID + 0;

    // edgeIDs
    newHalfedges[0]->edgeID = 2 * edgeID + (halfedgeID > twinID ? 0 : 1);
    newHalfedges[1]->edgeID = 2 * edgeCount + halfedgeID;
    newHalfedges[2]->edgeID = 2 * edgeCount + prevID;
    newHalfedges[3]->edgeID = 2 * prevEdgeID + (prevID > prevTwinID ? 1 : 0);

    // vertexIDs
    newHalfedges[0]->vertexID = vertexID;
    newHalfedges[1]->vertexID = vertexCount + faceCount + edgeID;
    newHalfedges[2]->vertexID = vertexCount + faceID;
    newHalfedges[3]->vertexID = vertexCount + faceCount + prevEdgeID;
}


void ccs__RefineCageHalfedges(cc_Subd *subd)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t vertexCount = ccm_VertexCount(cage);
    const int32_t edgeCount = ccm_EdgeCount(cage);
    const int32_t faceCount = ccm_FaceCount(cage);
    const int32_t halfedgeCount = ccm_HalfedgeCount(cage);
    cc_Halfedge_SemiRegular *halfedgesOut = subd->halfedges;
    int32_t intermed = (halfedgeCount + NUM_THREADS - 1) / NUM_THREADS;

    printf("halfedgeCount %d, num_blocks %d, num_threads %d \n", halfedgeCount, intermed, NUM_THREADS);
    RefineCageInner<<<EACH_ELEM(halfedgeCount)>>>(cage, vertexCount, edgeCount, faceCount, halfedgeCount, halfedgesOut);
    cudaDeviceSynchronize();
}

__global__ void RefineInnerHalfedges(cc_Subd *subd, int32_t depth, const cc_Mesh *cage, int32_t halfedgeCount, int32_t vertexCount, int32_t edgeCount, int32_t faceCount, int32_t stride, cc_Halfedge_SemiRegular *halfedgesOut){
    CHECK_TID(halfedgeCount)
    int32_t halfedgeID = TID;
    const int32_t twinID = ccs_HalfedgeTwinID(subd, halfedgeID, depth);
    const int32_t prevID = ccm_HalfedgePrevID_Quad(halfedgeID);
    const int32_t nextID = ccm_HalfedgeNextID_Quad(halfedgeID);
    const int32_t faceID = ccm_HalfedgeFaceID_Quad(halfedgeID);
    const int32_t edgeID = ccs_HalfedgeEdgeID(subd, halfedgeID, depth);
    const int32_t vertexID = ccs_HalfedgeVertexID(subd, halfedgeID, depth);
    const int32_t prevEdgeID = ccs_HalfedgeEdgeID(subd, prevID, depth);
    const int32_t prevTwinID = ccs_HalfedgeTwinID(subd, prevID, depth);
    const int32_t twinNextID = ccm_HalfedgeNextID_Quad(twinID);
    cc_Halfedge_SemiRegular *newHalfedges[4] = {
        &halfedgesOut[(4 * halfedgeID + 0)],
        &halfedgesOut[(4 * halfedgeID + 1)],
        &halfedgesOut[(4 * halfedgeID + 2)],
        &halfedgesOut[(4 * halfedgeID + 3)]
    };

    // twinIDs
    newHalfedges[0]->twinID = 4 * twinNextID + 3;
    newHalfedges[1]->twinID = 4 * nextID     + 2;
    newHalfedges[2]->twinID = 4 * prevID     + 1;
    newHalfedges[3]->twinID = 4 * prevTwinID + 0;

    // edgeIDs
    newHalfedges[0]->edgeID = 2 * edgeID + (halfedgeID > twinID ? 0 : 1);
    newHalfedges[1]->edgeID = 2 * edgeCount + halfedgeID;
    newHalfedges[2]->edgeID = 2 * edgeCount + prevID;
    newHalfedges[3]->edgeID = 2 * prevEdgeID + (prevID > prevTwinID ? 1 : 0);

    // vertexIDs
    newHalfedges[0]->vertexID = vertexID;
    newHalfedges[1]->vertexID = vertexCount + faceCount + edgeID;
    newHalfedges[2]->vertexID = vertexCount + faceID;
    newHalfedges[3]->vertexID = vertexCount + faceCount + prevEdgeID;
}


/*******************************************************************************
 * RefineHalfedges -- Applies halfedge refinement on the subd
 *
 * This routine computes the halfedges of the next subd level.
 *
 */
static void ccs__RefineHalfedges(cc_Subd *subd, int32_t depth)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t halfedgeCount = ccm_HalfedgeCountAtDepth(cage, depth);
    const int32_t vertexCount = ccm_VertexCountAtDepth_Fast(cage, depth);
    const int32_t edgeCount = ccm_EdgeCountAtDepth_Fast(cage, depth);
    const int32_t faceCount = ccm_FaceCountAtDepth_Fast(cage, depth);
    const int32_t stride = ccs_CumulativeHalfedgeCountAtDepth(cage, depth);
    cc_Halfedge_SemiRegular *halfedgesOut = &subd->halfedges[stride];
    RefineInnerHalfedges<<<EACH_ELEM(halfedgeCount)>>>(subd, depth, cage, halfedgeCount, vertexCount, edgeCount, faceCount, stride, halfedgesOut);
    cudaDeviceSynchronize();
}


/*******************************************************************************
 * RefineHalfedges
 *
 */
void ccs_RefineHalfedges(cc_Subd *subd)
{
    printf("Code has changed to Global Call\n");
    const int32_t maxDepth = ccs_MaxDepth(subd);

    ccs__RefineCageHalfedges(subd);

    for (int32_t depth = 1; depth < maxDepth; ++depth) {
        ccs__RefineHalfedges(subd, depth);
    }
}


#ifndef CC_DISABLE_UV
/*******************************************************************************
 * RefineCageVertexUvs -- Refines UVs of the cage mesh
 *
 * This routine computes the UVs of the control cage after one subdivision
 * step and stores them in the subd. Note that since UVs are not linked to
 * the topology of the mesh, we store the results of the UV computation
 * within the halfedge buffer.
 *
 */
__global__ void ccs__RefineCageVertexUvs(cc_Subd *subd)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t halfedgeCount = ccm_HalfedgeCount(cage);
    cc_Halfedge_SemiRegular *halfedgesOut = subd->halfedges;

    int edges_per_thread = std::ceil(float(halfedgeCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t halfedgeID = start; halfedgeID < end && halfedgeID < halfedgeCount; ++halfedgeID) {
        const int32_t prevID = ccm_HalfedgePrevID(cage, halfedgeID);
        const int32_t nextID = ccm_HalfedgeNextID(cage, halfedgeID);
        const cc_VertexUv uv = ccm_HalfedgeVertexUv(cage, halfedgeID);
        const cc_VertexUv nextUv = ccm_HalfedgeVertexUv(cage, nextID);
        const cc_VertexUv prevUv = ccm_HalfedgeVertexUv(cage, prevID);
        cc_VertexUv edgeUv, prevEdgeUv;
        cc_VertexUv faceUv = uv;
        int32_t m = 1;
        cc_Halfedge_SemiRegular *newHalfedges[4] = {
            &halfedgesOut[(4 * halfedgeID + 0)],
            &halfedgesOut[(4 * halfedgeID + 1)],
            &halfedgesOut[(4 * halfedgeID + 2)],
            &halfedgesOut[(4 * halfedgeID + 3)]
        };

        cc__Lerp2f(edgeUv.array    , uv.array, nextUv.array, 0.5f);
        cc__Lerp2f(prevEdgeUv.array, uv.array, prevUv.array, 0.5f);

        for (int32_t halfedgeIt = ccm_HalfedgeNextID(cage, halfedgeID);
                     halfedgeIt != halfedgeID;
                     halfedgeIt = ccm_HalfedgeNextID(cage, halfedgeIt)) {
            const cc_VertexUv uv = ccm_HalfedgeVertexUv(cage, halfedgeIt);

            faceUv.u+= uv.array[0];
            faceUv.v+= uv.array[1];
            ++m;
        }
        faceUv.u/= (float)m;
        faceUv.v/= (float)m;

        newHalfedges[0]->uvID = cc__EncodeUv(uv);
        newHalfedges[1]->uvID = cc__EncodeUv(edgeUv);
        newHalfedges[2]->uvID = cc__EncodeUv(faceUv);
        newHalfedges[3]->uvID = cc__EncodeUv(prevEdgeUv);
    }
    __syncthreads();
}


/*******************************************************************************
 * RefineVertexUvs -- Applies UV refinement on the subd
 *
 * This routine computes the UVs of the next subd level.
 *
 */
__global__ void ccs__RefineVertexUvs(cc_Subd *subd, int32_t depth)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t halfedgeCount = ccm_HalfedgeCountAtDepth(cage, depth);
    const int32_t stride = ccs_CumulativeHalfedgeCountAtDepth(cage, depth);
    cc_Halfedge_SemiRegular *halfedgesOut = &subd->halfedges[stride];

    int edges_per_thread = std::ceil(float(halfedgeCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t halfedgeID = start; halfedgeID < end && halfedgeID < halfedgeCount; ++halfedgeID) {
        const int32_t prevID = ccm_HalfedgePrevID_Quad(halfedgeID);
        const int32_t nextID = ccm_HalfedgeNextID_Quad(halfedgeID);
        const cc_VertexUv uv = ccs_HalfedgeVertexUv(subd, halfedgeID, depth);
        const cc_VertexUv nextUv = ccs_HalfedgeVertexUv(subd, nextID, depth);
        const cc_VertexUv prevUv = ccs_HalfedgeVertexUv(subd, prevID, depth);
        cc_VertexUv edgeUv, prevEdgeUv;
        cc_VertexUv faceUv = uv;
        cc_Halfedge_SemiRegular *newHalfedges[4] = {
            &halfedgesOut[(4 * halfedgeID + 0)],
            &halfedgesOut[(4 * halfedgeID + 1)],
            &halfedgesOut[(4 * halfedgeID + 2)],
            &halfedgesOut[(4 * halfedgeID + 3)]
        };

        cc__Lerp2f(edgeUv.array    , uv.array, nextUv.array, 0.5f);
        cc__Lerp2f(prevEdgeUv.array, uv.array, prevUv.array, 0.5f);

        for (int32_t halfedgeIt = ccs_HalfedgeNextID(subd, halfedgeID, depth);
                     halfedgeIt != halfedgeID;
                     halfedgeIt = ccs_HalfedgeNextID(subd, halfedgeIt, depth)) {
            const cc_VertexUv uv = ccs_HalfedgeVertexUv(subd, halfedgeIt, depth);

            faceUv.u+= uv.array[0];
            faceUv.v+= uv.array[1];
        }
        faceUv.u/= 4.0f;
        faceUv.v/= 4.0f;

        newHalfedges[0]->uvID = ccs__HalfedgeVertexUvID(subd, halfedgeID, depth);
        newHalfedges[1]->uvID = cc__EncodeUv(edgeUv);
        newHalfedges[2]->uvID = cc__EncodeUv(faceUv);
        newHalfedges[3]->uvID = cc__EncodeUv(prevEdgeUv);
    }
    __syncthreads();
}


/*******************************************************************************
 * RefineUvs
 *
 */
 void ccs_RefineVertexUvs(cc_Subd *subd)
{
    if (ccm_UvCount(subd->cage) > 0) {
        const int32_t maxDepth = ccs_MaxDepth(subd);

        ccs__RefineCageVertexUvs<<<1, NUM_THREADS>>>(subd);

        for (int32_t depth = 1; depth < maxDepth; ++depth) {
            ccs__RefineVertexUvs<<<1, NUM_THREADS>>>(subd, depth);
        }
    }
}
#endif


/*******************************************************************************
 * RefineCageCreases -- Applies crease subdivision on the cage mesh
 *
 * This routine computes the creases of the control cage after one subdivision
 * step and stores them in the subd.
 *
 */
__global__ void ccs__RefineCageCreases(cc_Subd *subd)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t edgeCount = cage->edgeCount;
    cc_Crease *creasesOut = subd->creases;

    int edges_per_thread = std::ceil(float(edgeCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;
    for (int32_t edgeID = start; edgeID < end && edgeID < edgeCount; ++edgeID) {
        const int32_t nextID = ccm_CreaseNextID(cage, edgeID);
        const int32_t prevID = ccm_CreasePrevID(cage, edgeID);
        const bool t1 = ccm_CreasePrevID(cage, nextID) == edgeID && nextID != edgeID;
        const bool t2 = ccm_CreaseNextID(cage, prevID) == edgeID && prevID != edgeID;
        const float thisS = 3.0f * ccm_CreaseSharpness(cage, edgeID);
        const float nextS = ccm_CreaseSharpness(cage, nextID);
        const float prevS = ccm_CreaseSharpness(cage, prevID);
        cc_Crease *newCreases[2] = {
            &creasesOut[(2 * edgeID + 0)],
            &creasesOut[(2 * edgeID + 1)]
        };

        // next rule
        newCreases[0]->nextID = 2 * edgeID + 1;
        newCreases[1]->nextID = 2 * nextID + (t1 ? 0 : 1);

        // prev rule
        newCreases[0]->prevID = 2 * prevID + (t2 ? 1 : 0);
        newCreases[1]->prevID = 2 * edgeID + 0;

        // sharpness rule
        newCreases[0]->sharpness = cc__Maxf(0.0f, (prevS + thisS) / 4.0f - 1.0f);
        newCreases[1]->sharpness = cc__Maxf(0.0f, (thisS + nextS) / 4.0f - 1.0f);
    }
    __syncthreads();
}


/*******************************************************************************
 * RefineCreases -- Applies crease subdivision on the subd
 *
 * This routine computes the topology of the next subd level.
 *
 */
__global__ void ccs__RefineCreases(cc_Subd *subd, int32_t depth)
{
    const cc_Mesh *cage = subd->cage;
    const int32_t creaseCount = ccm_CreaseCountAtDepth(cage, depth);
    const int32_t stride = ccs_CumulativeCreaseCountAtDepth(cage, depth);
    cc_Crease *creasesOut = &subd->creases[stride];

    int edges_per_thread = std::ceil(float(creaseCount) / float(NUM_THREADS));
    int start = threadIdx.x;
    int end = threadIdx.x + edges_per_thread;

    for (int32_t edgeID = start; edgeID < end && edgeID < creaseCount ; ++edgeID) {
        const int32_t nextID = ccs_CreaseNextID_Fast(subd, edgeID, depth);
        const int32_t prevID = ccs_CreasePrevID_Fast(subd, edgeID, depth);
        const bool t1 = ccs_CreasePrevID_Fast(subd, nextID, depth) == edgeID && nextID != edgeID;
        const bool t2 = ccs_CreaseNextID_Fast(subd, prevID, depth) == edgeID && prevID != edgeID;
        const float thisS = 3.0f * ccs_CreaseSharpness_Fast(subd, edgeID, depth);
        const float nextS = ccs_CreaseSharpness_Fast(subd, nextID, depth);
        const float prevS = ccs_CreaseSharpness_Fast(subd, prevID, depth);
        cc_Crease *newCreases[2] = {
            &creasesOut[(2 * edgeID + 0)],
            &creasesOut[(2 * edgeID + 1)]
        };

        // next rule
        newCreases[0]->nextID = 2 * edgeID + 1;
        newCreases[1]->nextID = 2 * nextID + (t1 ? 0 : 1);

        // prev rule
        newCreases[0]->prevID = 2 * prevID + (t2 ? 1 : 0);
        newCreases[1]->prevID = 2 * edgeID + 0;

        // sharpness rule
        newCreases[0]->sharpness = cc__Maxf(0.0f, (prevS + thisS) / 4.0f - 1.0f);
        newCreases[1]->sharpness = cc__Maxf(0.0f, (thisS + nextS) / 4.0f - 1.0f);
    }
    __syncthreads();
}


/*******************************************************************************
 * RefineCreases
 *
 */
 void ccs_RefineCreases(cc_Subd *subd)
{
    const int32_t maxDepth = ccs_MaxDepth(subd);

    ccs__RefineCageCreases<<<1, NUM_THREADS>>>(subd);

    for (int32_t depth = 1; depth < maxDepth; ++depth) {
        ccs__RefineCreases<<<1, NUM_THREADS>>>(subd, depth);
        cudaDeviceSynchronize();
    }
}


/*******************************************************************************
 * Refine -- Computes and stores the result of Catmull Clark subdivision.
 *
 * The subdivision is computed down to the maxDepth parameter.
 *
 */
void ccs__RefineTopology(cc_Subd *subd)
{
    ccs_RefineHalfedges(subd);
    ccs_RefineCreases(subd);
#ifndef CC_DISABLE_UV
    ccs_RefineVertexUvs(subd);
#endif
}

 void ccs_Refine_Scatter(cc_Subd *subd)
{
    ccs__RefineTopology(subd);
    ccs_RefineVertexPoints_Scatter(subd);
}

 void ccs_Refine_Gather(cc_Subd *subd)
{
    ccs__RefineTopology(subd);
    ccs_RefineVertexPoints_Gather(subd);
}

 void ccs_Refine_NoCreases_Scatter(cc_Subd *subd)
{
    ccs__RefineTopology(subd);
    ccs_RefineVertexPoints_NoCreases_Scatter(subd);
}

 void ccs_Refine_NoCreases_Gather(cc_Subd *subd)
{
    ccs__RefineTopology(subd);
    ccs_RefineVertexPoints_NoCreases_Gather(subd);
}