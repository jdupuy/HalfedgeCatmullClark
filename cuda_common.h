#define CCDEF __device__
#include <cuda.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

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
#    define CC_MALLOC(x) (cudaMalloc(x))
#    define CC_FREE(x) (cudaFree(x))
#else
#    ifndef CC_FREE
#        error CC_MALLOC defined without CC_FREE
#    endif
#endif

#ifndef CC_MEMCPY
#    define CC_MEMCPY(dest, src, count) cudaMemcpy(dest, src, count)
#endif

#ifndef CC_MEMSET
#    define CC_MEMSET(ptr, value, num) cudaMemset(ptr, value, num)
#endif


// point data
typedef union {
    struct {float x, y, z;};
    float array[3];
} cc_VertexPoint;

// uv data
typedef union {
    struct {float u, v;};
    float array[2];
} cc_VertexUv;

// crease data
typedef struct {
    int32_t nextID;
    int32_t prevID;
    float sharpness;
} cc_Crease;

// generic halfedge data
typedef struct {
    int32_t twinID;
    int32_t nextID;
    int32_t prevID;
    int32_t faceID;
    int32_t edgeID;
    int32_t vertexID;
    int32_t uvID;
} cc_Halfedge;

// specialized halfedge data for semi-regular (e.g., quad-only) meshes
typedef struct {
    int32_t twinID;
    int32_t edgeID;
    int32_t vertexID;
#ifndef CC_DISABLE_UV
    int32_t uvID;
#endif
} cc_Halfedge_SemiRegular;

// mesh data-structure
typedef struct {
    int32_t vertexCount;
    int32_t uvCount;
    int32_t halfedgeCount;
    int32_t edgeCount;
    int32_t faceCount;
    int32_t *vertexToHalfedgeIDs;
    int32_t *edgeToHalfedgeIDs;
    int32_t *faceToHalfedgeIDs;
    cc_VertexPoint *vertexPoints;
    cc_VertexUv *uvs;
    cc_Halfedge *halfedges;
    cc_Crease *creases;
} cc_Mesh;

// ctor / dtor
CCDEF cc_Mesh *ccm_Load(const char *filename);
CCDEF cc_Mesh *ccm_Create(int32_t vertexCount,
                          int32_t uvCount,
                          int32_t halfedgeCount,
                          int32_t edgeCount,
                          int32_t faceCount);
CCDEF void ccm_Release(cc_Mesh *mesh);

// export
CCDEF bool ccm_Save(const cc_Mesh *mesh, const char *filename);

// count queries
CCDEF int32_t ccm_FaceCount(const cc_Mesh *mesh);
CCDEF int32_t ccm_EdgeCount(const cc_Mesh *mesh);
CCDEF int32_t ccm_HalfedgeCount(const cc_Mesh *mesh);
CCDEF int32_t ccm_CreaseCount(const cc_Mesh *mesh);
CCDEF int32_t ccm_VertexCount(const cc_Mesh *mesh);
CCDEF int32_t ccm_UvCount(const cc_Mesh *mesh);

// counts at a given Catmull-Clark subdivision depth
CCDEF int32_t ccm_HalfedgeCountAtDepth(const cc_Mesh *cage, int32_t depth);
CCDEF int32_t ccm_CreaseCountAtDepth(const cc_Mesh *cage, int32_t depth);
CCDEF int32_t ccm_FaceCountAtDepth     (const cc_Mesh *cage, int32_t depth);
CCDEF int32_t ccm_FaceCountAtDepth_Fast(const cc_Mesh *cage, int32_t depth);
CCDEF int32_t ccm_EdgeCountAtDepth     (const cc_Mesh *cage, int32_t depth);
CCDEF int32_t ccm_EdgeCountAtDepth_Fast(const cc_Mesh *cage, int32_t depth);
CCDEF int32_t ccm_VertexCountAtDepth     (const cc_Mesh *cage, int32_t depth);
CCDEF int32_t ccm_VertexCountAtDepth_Fast(const cc_Mesh *cage, int32_t depth);

// data-access (O(1))
CCDEF int32_t ccm_HalfedgeTwinID(const cc_Mesh *mesh, int32_t halfedgeID);
CCDEF int32_t ccm_HalfedgeNextID(const cc_Mesh *mesh, int32_t halfedgeID);
CCDEF int32_t ccm_HalfedgePrevID(const cc_Mesh *mesh, int32_t halfedgeID);
CCDEF int32_t ccm_HalfedgeFaceID(const cc_Mesh *mesh, int32_t halfedgeID);
CCDEF int32_t ccm_HalfedgeEdgeID(const cc_Mesh *mesh, int32_t halfedgeID);
CCDEF int32_t ccm_HalfedgeVertexID(const cc_Mesh *mesh, int32_t halfedgeID);
CCDEF int32_t ccm_HalfedgeUvID(const cc_Mesh *mesh, int32_t halfedgeID);
CCDEF float ccm_HalfedgeSharpness(const cc_Mesh *mesh, int32_t halfedgeID);
CCDEF cc_VertexPoint ccm_HalfedgeVertexPoint(const cc_Mesh *mesh, int32_t halfedgeID);
CCDEF cc_VertexUv ccm_HalfedgeVertexUv(const cc_Mesh *mesh, int32_t halfedgeID);
CCDEF int32_t ccm_CreaseNextID(const cc_Mesh *mesh, int32_t edgeID);
CCDEF int32_t ccm_CreasePrevID(const cc_Mesh *mesh, int32_t edgeID);
CCDEF float ccm_CreaseSharpness(const cc_Mesh *mesh, int32_t edgeID);
CCDEF cc_VertexPoint ccm_VertexPoint(const cc_Mesh *mesh, int32_t vertexID);
CCDEF cc_VertexUv ccm_Uv(const cc_Mesh *mesh, int32_t uvID);
CCDEF int32_t ccm_HalfedgeNextID_Quad(int32_t halfedgeID);
CCDEF int32_t ccm_HalfedgePrevID_Quad(int32_t halfedgeID);
CCDEF int32_t ccm_HalfedgeFaceID_Quad(int32_t halfedgeID);

// (vertex, edge, face) -> halfedge mappings (O(1))
CCDEF int32_t ccm_VertexToHalfedgeID(const cc_Mesh *mesh, int32_t vertexID);
CCDEF int32_t ccm_EdgeToHalfedgeID(const cc_Mesh *mesh, int32_t edgeID);
CCDEF int32_t ccm_FaceToHalfedgeID(const cc_Mesh *mesh, int32_t faceID);
CCDEF int32_t ccm_FaceToHalfedgeID_Quad(int32_t faceID);

// halfedge remappings (O(1))
CCDEF int32_t ccm_NextVertexHalfedgeID(const cc_Mesh *mesh, int32_t halfedgeID);
CCDEF int32_t ccm_PrevVertexHalfedgeID(const cc_Mesh *mesh, int32_t halfedgeID);

// subdivision surface API

// subd data-structure
typedef struct {
    const cc_Mesh *cage;
    cc_VertexPoint *vertexPoints;
    cc_Halfedge_SemiRegular *halfedges;
    cc_Crease *creases;
    int32_t maxDepth;
} cc_Subd;

/*******************************************************************************
 * Header File Data Structure
 *
 * This represents the header we use to uniquely identify the cc_Mesh files
 * and provide the fundamental information to properly decode the rest of the
 * file.
 *
 */
typedef struct {
    int64_t magic;
    int32_t vertexCount;
    int32_t uvCount;
    int32_t halfedgeCount;
    int32_t edgeCount;
    int32_t faceCount;
} ccm__Header;



// ctor / dtor
CCDEF cc_Subd *ccs_Create(const cc_Mesh *cage, int32_t maxDepth);
CCDEF void ccs_Release(cc_Subd *subd);

// subd queries
CCDEF int32_t ccs_MaxDepth(const cc_Subd *subd);
CCDEF int32_t ccs_VertexCount(const cc_Subd *subd);
CCDEF int32_t ccs_CumulativeFaceCount(const cc_Subd *subd);
CCDEF int32_t ccs_CumulativeEdgeCount(const cc_Subd *subd);
CCDEF int32_t ccs_CumulativeCreaseCount(const cc_Subd *subd);
CCDEF int32_t ccs_CumulativeVertexCount(const cc_Subd *subd);
CCDEF int32_t ccs_CumulativeHalfedgeCount(const cc_Subd *subd);
CCDEF int32_t ccs_CumulativeHalfedgeCountAtDepth(const cc_Mesh *cage, int32_t depth);
CCDEF int32_t ccs_CumulativeVertexCountAtDepth(const cc_Mesh *cage, int32_t depth);
CCDEF int32_t ccs_CumulativeFaceCountAtDepth(const cc_Mesh *cage, int32_t depth);
CCDEF int32_t ccs_CumulativeEdgeCountAtDepth(const cc_Mesh *cage, int32_t depth);
CCDEF int32_t ccs_CumulativeCreaseCountAtDepth(const cc_Mesh *cage, int32_t depth);

// O(1) data-access
CCDEF int32_t ccs_HalfedgeTwinID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
CCDEF int32_t ccs_HalfedgeNextID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
CCDEF int32_t ccs_HalfedgePrevID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
CCDEF int32_t ccs_HalfedgeFaceID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
CCDEF int32_t ccs_HalfedgeEdgeID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
CCDEF int32_t ccs_HalfedgeVertexID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
CCDEF cc_VertexPoint ccs_HalfedgeVertexPoint(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
#ifndef CC_DISABLE_UV
CCDEF cc_VertexUv ccs_HalfedgeVertexUv(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
#endif
CCDEF float ccs_HalfedgeSharpness   (const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
CCDEF int32_t ccs_CreaseNextID_Fast (const cc_Subd *subd, int32_t edgeID, int32_t depth);
CCDEF int32_t ccs_CreaseNextID      (const cc_Subd *subd, int32_t edgeID, int32_t depth);
CCDEF int32_t ccs_CreasePrevID_Fast (const cc_Subd *subd, int32_t edgeID, int32_t depth);
CCDEF int32_t ccs_CreasePrevID      (const cc_Subd *subd, int32_t edgeID, int32_t depth);
CCDEF float ccs_CreaseSharpness_Fast(const cc_Subd *subd, int32_t edgeID, int32_t depth);
CCDEF float ccs_CreaseSharpness     (const cc_Subd *subd, int32_t edgeID, int32_t depth);
CCDEF cc_VertexPoint ccs_VertexPoint(const cc_Subd *subd, int32_t vertexID, int32_t depth);

// halfedge remapping (O(1))
CCDEF int32_t ccs_NextVertexHalfedgeID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
CCDEF int32_t ccs_PrevVertexHalfedgeID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);

// (vertex, edge, face) -> halfedge mappings
CCDEF int32_t ccs_VertexToHalfedgeID(const cc_Subd *subd,
                                     int32_t vertexID,
                                     int32_t depth);
CCDEF int32_t ccs_EdgeToHalfedgeID(const cc_Subd *mesh,
                                   int32_t edgeID,
                                   int32_t depth);
CCDEF int32_t ccs_FaceToHalfedgeID(const cc_Subd *mesh,
                                   int32_t faceID,
                                   int32_t depth);

// (re-)compute catmull clark subdivision
__global__ void ccs_Refine_Gather(cc_Subd *subd);
__global__ void ccs_Refine_Scatter(cc_Subd *subd);
__global__ void ccs_RefineVertexPoints_Gather(cc_Subd *subd);
__global__ void ccs_RefineVertexPoints_Scatter(cc_Subd *subd);
__global__ void ccs_RefineHalfedges(cc_Subd *subd);
__global__ void ccs_RefineCreases(cc_Subd *subd);
#ifndef CC_DISABLE_UV
__global__ void ccs_RefineVertexUvs(cc_Subd *subd);
#endif

// (re-)compute catmull clark vertex points without semi-sharp creases
__global__ void ccs_Refine_NoCreases_Gather(cc_Subd *subd);
__global__ void ccs_Refine_NoCreases_Scatter(cc_Subd *subd);
__global__ void ccs_RefineVertexPoints_NoCreases_Gather(cc_Subd *subd);
__global__ void ccs_RefineVertexPoints_NoCreases_Scatter(cc_Subd *subd);
