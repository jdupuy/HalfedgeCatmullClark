#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>
#include "Utilities.h"

// point data
typedef union {
    struct {float x, y, z;};
    float array[3];
} cc_VertexPoint;


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

// subd data-structure
typedef struct {
    const cc_Mesh *cage;
    cc_VertexPoint *vertexPoints;
    cc_Halfedge_SemiRegular *halfedges;
    cc_Crease *creases;
    int32_t maxDepth;
} cc_Subd;

// ctor / dtor
cc_Subd *ccs_Create(const cc_Mesh *cage, int32_t maxDepth);
 void ccs_Release(cc_Subd *subd);



/***************
 SUBD STUFF
****************/
 int32_t ccs_MaxDepth(const cc_Subd *subd);
 int32_t ccs_VertexCount(const cc_Subd *subd);
 int32_t ccs_CumulativeFaceCount(const cc_Subd *subd);
 int32_t ccs_CumulativeEdgeCount(const cc_Subd *subd);
 int32_t ccs_CumulativeCreaseCount(const cc_Subd *subd);
 int32_t ccs_CumulativeVertexCount(const cc_Subd *subd);
 int32_t ccs_CumulativeHalfedgeCount(const cc_Subd *subd);
 int32_t ccs_CumulativeHalfedgeCountAtDepth(const cc_Mesh *cage, int32_t depth);
 int32_t ccs_CumulativeVertexCountAtDepth(const cc_Mesh *cage, int32_t depth);
 int32_t ccs_CumulativeFaceCountAtDepth(const cc_Mesh *cage, int32_t depth);
 int32_t ccs_CumulativeEdgeCountAtDepth(const cc_Mesh *cage, int32_t depth);
 int32_t ccs_CumulativeCreaseCountAtDepth(const cc_Mesh *cage, int32_t depth);

// O(1) data-access
 int32_t ccs_HalfedgeTwinID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
 int32_t ccs_HalfedgeNextID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
 int32_t ccs_HalfedgePrevID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
 int32_t ccs_HalfedgeFaceID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
 int32_t ccs_HalfedgeEdgeID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
 int32_t ccs_HalfedgeVertexID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
 cc_VertexPoint ccs_HalfedgeVertexPoint(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
#ifndef CC_DISABLE_UV
 cc_VertexUv ccs_HalfedgeVertexUv(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
#endif
 float ccs_HalfedgeSharpness   (const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
 int32_t ccs_CreaseNextID_Fast (const cc_Subd *subd, int32_t edgeID, int32_t depth);
 int32_t ccs_CreaseNextID      (const cc_Subd *subd, int32_t edgeID, int32_t depth);
 int32_t ccs_CreasePrevID_Fast (const cc_Subd *subd, int32_t edgeID, int32_t depth);
 int32_t ccs_CreasePrevID      (const cc_Subd *subd, int32_t edgeID, int32_t depth);
 float ccs_CreaseSharpness_Fast(const cc_Subd *subd, int32_t edgeID, int32_t depth);
 float ccs_CreaseSharpness     (const cc_Subd *subd, int32_t edgeID, int32_t depth);
 cc_VertexPoint ccs_VertexPoint(const cc_Subd *subd, int32_t vertexID, int32_t depth);

// halfedge remapping (O(1))
 int32_t ccs_NextVertexHalfedgeID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
 int32_t ccs_PrevVertexHalfedgeID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);

// (vertex, edge, face) -> halfedge mappings
 int32_t ccs_VertexToHalfedgeID(const cc_Subd *subd,
                                     int32_t vertexID,
                                     int32_t depth);
 int32_t ccs_EdgeToHalfedgeID(const cc_Subd *mesh,
                                   int32_t edgeID,
                                   int32_t depth);
 int32_t ccs_FaceToHalfedgeID(const cc_Subd *mesh,
                                   int32_t faceID,
                                   int32_t depth);

/***************
 MESH Creation
****************/
// ctor / dtor
 cc_Mesh *ccm_Load(const char *filename);
 cc_Mesh *ccm_Create(int32_t vertexCount,
                          int32_t uvCount,
                          int32_t halfedgeCount,
                          int32_t edgeCount,
                          int32_t faceCount);
 void ccm_Release(cc_Mesh *mesh);

// export
 bool ccm_Save(const cc_Mesh *mesh, const char *filename);


/* **************
 MESH Counting
*************** */
// count queries
 int32_t ccm_FaceCount(const cc_Mesh *mesh);
 int32_t ccm_EdgeCount(const cc_Mesh *mesh);
 int32_t ccm_HalfedgeCount(const cc_Mesh *mesh);
 int32_t ccm_CreaseCount(const cc_Mesh *mesh);
 int32_t ccm_VertexCount(const cc_Mesh *mesh);
 int32_t ccm_UvCount(const cc_Mesh *mesh);


/***************
Mesh Counting at Subdivision Depth
****************/
// counts at a given Catmull-Clark subdivision depth
 int32_t ccm_HalfedgeCountAtDepth(const cc_Mesh *cage, int32_t depth);
 int32_t ccm_CreaseCountAtDepth(const cc_Mesh *cage, int32_t depth);
 int32_t ccm_FaceCountAtDepth     (const cc_Mesh *cage, int32_t depth);
 int32_t ccm_FaceCountAtDepth_Fast(const cc_Mesh *cage, int32_t depth);
 int32_t ccm_EdgeCountAtDepth     (const cc_Mesh *cage, int32_t depth);
 int32_t ccm_EdgeCountAtDepth_Fast(const cc_Mesh *cage, int32_t depth);
 int32_t ccm_VertexCountAtDepth     (const cc_Mesh *cage, int32_t depth);
 int32_t ccm_VertexCountAtDepth_Fast(const cc_Mesh *cage, int32_t depth);

/***************
 Data Access at O(1)
****************/
 int32_t ccm_HalfedgeTwinID(const cc_Mesh *mesh, int32_t halfedgeID);
 int32_t ccm_HalfedgeNextID(const cc_Mesh *mesh, int32_t halfedgeID);
 int32_t ccm_HalfedgePrevID(const cc_Mesh *mesh, int32_t halfedgeID);
 int32_t ccm_HalfedgeFaceID(const cc_Mesh *mesh, int32_t halfedgeID);
 int32_t ccm_HalfedgeEdgeID(const cc_Mesh *mesh, int32_t halfedgeID);
 int32_t ccm_HalfedgeVertexID(const cc_Mesh *mesh, int32_t halfedgeID);
 int32_t ccm_HalfedgeUvID(const cc_Mesh *mesh, int32_t halfedgeID);
 float ccm_HalfedgeSharpness(const cc_Mesh *mesh, int32_t halfedgeID);
 cc_VertexPoint ccm_HalfedgeVertexPoint(const cc_Mesh *mesh, int32_t halfedgeID);
 cc_VertexUv ccm_HalfedgeVertexUv(const cc_Mesh *mesh, int32_t halfedgeID);
 int32_t ccm_CreaseNextID(const cc_Mesh *mesh, int32_t edgeID);
 int32_t ccm_CreasePrevID(const cc_Mesh *mesh, int32_t edgeID);
 float ccm_CreaseSharpness(const cc_Mesh *mesh, int32_t edgeID);
 cc_VertexPoint ccm_VertexPoint(const cc_Mesh *mesh, int32_t vertexID);
 cc_VertexUv ccm_Uv(const cc_Mesh *mesh, int32_t uvID);
 int32_t ccm_HalfedgeNextID_Quad(int32_t halfedgeID);
 int32_t ccm_HalfedgePrevID_Quad(int32_t halfedgeID);
 int32_t ccm_HalfedgeFaceID_Quad(int32_t halfedgeID);


/***************
(vertex, edge, face) -> halfedge mappings (O(1))
****************/
 int32_t ccm_VertexToHalfedgeID(const cc_Mesh *mesh, int32_t vertexID);
 int32_t ccm_EdgeToHalfedgeID(const cc_Mesh *mesh, int32_t edgeID);
 int32_t ccm_FaceToHalfedgeID(const cc_Mesh *mesh, int32_t faceID);
 int32_t ccm_FaceToHalfedgeID_Quad(int32_t faceID);

// halfedge remappings (O(1))
 int32_t ccm_NextVertexHalfedgeID(const cc_Mesh *mesh, int32_t halfedgeID);
 int32_t ccm_PrevVertexHalfedgeID(const cc_Mesh *mesh, int32_t halfedgeID);

static const cc_Halfedge_SemiRegular *
ccs__Halfedge(const cc_Subd *subd, int32_t halfedgeID, int32_t depth)
{
    assert(depth <= ccs_MaxDepth(subd) && depth > 0);
    const int32_t stride = ccs_CumulativeHalfedgeCountAtDepth(subd->cage,
                                                              depth - 1);

    return &subd->halfedges[stride + halfedgeID];
}

/*******************************************************************************
 * Vertex to Halfedge Mapping
 *
 * This procedure returns the ID of one of the halfedge that connects a
 * given vertex. This routine has O(depth) complexity.
 *
 */
static int32_t
ccs__VertexToHalfedgeID_First(const cc_Mesh *cage, int32_t vertexID)
{
    const int32_t vertexCount = ccm_VertexCount(cage);
    const int32_t faceCount = ccm_FaceCount(cage);

    if /* [V + F, V + F + E) */ (vertexID >= vertexCount + faceCount) {
        const int32_t edgeID = vertexID - vertexCount - faceCount;

        return 4 * ccm_EdgeToHalfedgeID(cage, edgeID) + 1;

    } else if /* [V, V + F) */ (vertexID >= vertexCount) {
        const int32_t faceID = vertexID - vertexCount;

        return 4 * ccm_FaceToHalfedgeID(cage, faceID) + 2;

    } else /* [0, V) */ {

        return 4 * ccm_VertexToHalfedgeID(cage, vertexID) + 0;
    }
}

static cc_Halfedge *ccm__Halfedge(const cc_Mesh *mesh, int32_t halfedgeID)
{
    return &mesh->halfedges[halfedgeID];
}

static cc_Crease *ccm__Crease(const cc_Mesh *mesh, int32_t edgeID)
{
    return &mesh->creases[edgeID];
}

static int32_t
ccm__ScrollFaceHalfedgeID_Quad(int32_t halfedgeID, int32_t direction)
{
    const int32_t base = 3;
    const int32_t localID = (halfedgeID & base) + direction;

    return (halfedgeID & ~base) | (localID & base);
}

static const cc_Crease *
ccs__Crease(const cc_Subd *subd, int32_t edgeID, int32_t depth)
{
    assert(depth <= ccs_MaxDepth(subd) && depth > 0);
    const int32_t stride = ccs_CumulativeCreaseCountAtDepth(subd->cage,
                                                            depth - 1);

    return &subd->creases[stride + edgeID];
}

static int32_t ccs__EdgeToHalfedgeID_First(const cc_Mesh *cage, int32_t edgeID)
{
    const int32_t edgeCount = ccm_EdgeCount(cage);

    if /* [2E, 2E + H) */ (edgeID >= 2 * edgeCount) {
        const int32_t halfedgeID = edgeID - 2 * edgeCount;
        const int32_t nextID = ccm_HalfedgeNextID(cage, halfedgeID);

        return cc__Max(4 * halfedgeID + 1, 4 * nextID + 2);

    } else if /* */ ((edgeID & 1) == 1) {
        const int32_t halfedgeID = ccm_EdgeToHalfedgeID(cage, edgeID >> 1);
        const int32_t nextID = ccm_HalfedgeNextID(cage, halfedgeID);

        return 4 * nextID + 3;

    } else /* */ {
        const int32_t halfedgeID = ccm_EdgeToHalfedgeID(cage, edgeID >> 1);

        return 4 * halfedgeID + 0;
    }
}


static uint32_t
ccs__HalfedgeVertexUvID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth)
{
    return ccs__Halfedge(subd, halfedgeID, depth)->uvID;
}

