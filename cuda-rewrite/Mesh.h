#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include "Utilities.h"

// point data
typedef union {
    struct { float x, y, z;};
    float array[3];
} cc_VertexPoint_f;

// uv data
typedef union {
    struct {float u, v;};
    float array[2];
} cc_VertexUv_f;

// crease data
typedef struct {
    int32_t nextID;
    int32_t prevID;
    float sharpness;
} cc_Crease_f;

// point data
typedef union {
    struct {double x, y, z;};
    double array[3];
} cc_VertexPoint;


// crease data
typedef struct {
    int32_t nextID;
    int32_t prevID;
    double sharpness;
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

typedef struct {
    int64_t magic;
    int32_t vertexCount;
    int32_t uvCount;
    int32_t halfedgeCount;
    int32_t edgeCount;
    int32_t faceCount;
} ccm__Header;




/***************
 SUBD STUFF
****************/
 __host__ __device__ int32_t ccs_MaxDepth(const cc_Subd *subd);
 __host__ __device__ int32_t ccs_VertexCount(const cc_Subd *subd);
 __host__ __device__ int32_t ccs_CumulativeFaceCount(const cc_Subd *subd);
 __host__ __device__ int32_t ccs_CumulativeEdgeCount(const cc_Subd *subd);
 __host__ __device__ int32_t ccs_CumulativeCreaseCount(const cc_Subd *subd);
 __host__ __device__ int32_t ccs_CumulativeVertexCount(const cc_Subd *subd);
 __host__ __device__ int32_t ccs_CumulativeHalfedgeCount(const cc_Subd *subd);
 __host__ __device__ int32_t ccs_CumulativeHalfedgeCountAtDepth(const cc_Mesh *cage, int32_t depth);
 __host__ __device__ int32_t ccs_CumulativeVertexCountAtDepth(const cc_Mesh *cage, int32_t depth);
 __host__ __device__ int32_t ccs_CumulativeFaceCountAtDepth(const cc_Mesh *cage, int32_t depth);
 __host__ __device__ int32_t ccs_CumulativeEdgeCountAtDepth(const cc_Mesh *cage, int32_t depth);
 __host__ __device__ int32_t ccs_CumulativeCreaseCountAtDepth(const cc_Mesh *cage, int32_t depth);

// O(1) data-access
 __host__ __device__ int32_t ccs_HalfedgeTwinID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
 __host__ __device__ int32_t ccs_HalfedgeNextID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
 __host__ __device__ int32_t ccs_HalfedgePrevID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
 __host__ __device__ int32_t ccs_HalfedgeFaceID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
 __host__ __device__ int32_t ccs_HalfedgeEdgeID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
 __host__ __device__ int32_t ccs_HalfedgeVertexID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
 __host__ __device__ cc_VertexPoint ccs_HalfedgeVertexPoint(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
 __host__ __device__ cc_VertexUv ccs_HalfedgeVertexUv(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
 __host__ __device__ double ccs_HalfedgeSharpness   (const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
 __host__ __device__ int32_t ccs_CreaseNextID_Fast (const cc_Subd *subd, int32_t edgeID, int32_t depth);
 __host__ __device__ int32_t ccs_CreaseNextID      (const cc_Subd *subd, int32_t edgeID, int32_t depth);
 __host__ __device__ int32_t ccs_CreasePrevID_Fast (const cc_Subd *subd, int32_t edgeID, int32_t depth);
 __host__ __device__ int32_t ccs_CreasePrevID      (const cc_Subd *subd, int32_t edgeID, int32_t depth);
 __host__ __device__ double ccs_CreaseSharpness_Fast(const cc_Subd *subd, int32_t edgeID, int32_t depth);
 __host__ __device__ double ccs_CreaseSharpness     (const cc_Subd *subd, int32_t edgeID, int32_t depth);
  __host__ __device__ cc_VertexPoint ccs_VertexPoint(const cc_Subd *subd, int32_t vertexID, int32_t depth);

// halfedge remapping (O(1))
 __host__ __device__ int32_t ccs_NextVertexHalfedgeID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
 __host__ __device__ int32_t ccs_PrevVertexHalfedgeID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);

// (vertex, edge, face) -> halfedge mappings
 __host__ __device__ int32_t ccs_VertexToHalfedgeID(const cc_Subd *subd,
                                     int32_t vertexID,
                                     int32_t depth);
 __host__ __device__ int32_t ccs_EdgeToHalfedgeID(const cc_Subd *mesh,
                                   int32_t edgeID,
                                   int32_t depth);
 __host__ __device__ int32_t ccs_FaceToHalfedgeID(const cc_Subd *mesh,
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

 ccm__Header ccm__CreateHeader(const cc_Mesh *mesh);
 bool ccm__ReadHeader(FILE *stream, ccm__Header *header);
 bool ccm__ReadData(cc_Mesh *mesh, FILE *stream);
 int64_t ccm__Magic();


/* **************
 MESH Counting
*************** */
// count queries
 __host__ __device__ int32_t ccm_FaceCount(const cc_Mesh *mesh);
 __host__ __device__ int32_t ccm_EdgeCount(const cc_Mesh *mesh);
 __host__ __device__ int32_t ccm_HalfedgeCount(const cc_Mesh *mesh);
 __host__ __device__ int32_t ccm_CreaseCount(const cc_Mesh *mesh);
 __host__ __device__ int32_t ccm_VertexCount(const cc_Mesh *mesh);
 __host__ __device__ int32_t ccm_UvCount(const cc_Mesh *mesh);

/***************
Mesh Counting at Subdivision Depth
****************/
// counts at a given Catmull-Clark subdivision depth
 __host__ __device__ int32_t ccm_HalfedgeCountAtDepth(const cc_Mesh *cage, int32_t depth);
 __host__ __device__ int32_t ccm_CreaseCountAtDepth(const cc_Mesh *cage, int32_t depth);
 __host__ __device__ int32_t ccm_FaceCountAtDepth     (const cc_Mesh *cage, int32_t depth);
 __host__ __device__ int32_t ccm_FaceCountAtDepth_Fast(const cc_Mesh *cage, int32_t depth);
 __host__ __device__ int32_t ccm_EdgeCountAtDepth     (const cc_Mesh *cage, int32_t depth);
 __host__ __device__ int32_t ccm_EdgeCountAtDepth_Fast(const cc_Mesh *cage, int32_t depth);
 __host__ __device__ int32_t ccm_VertexCountAtDepth     (const cc_Mesh *cage, int32_t depth);
 __host__ __device__ int32_t ccm_VertexCountAtDepth_Fast(const cc_Mesh *cage, int32_t depth);

/***************
 Data Access at O(1)
****************/
 __host__ __device__ int32_t ccm_HalfedgeTwinID(const cc_Mesh *mesh, int32_t halfedgeID);
 __host__ __device__ int32_t ccm_HalfedgeNextID(const cc_Mesh *mesh, int32_t halfedgeID);
 __host__ __device__  int32_t ccm_HalfedgePrevID(const cc_Mesh *mesh, int32_t halfedgeID);
 __host__ __device__  int32_t ccm_HalfedgeFaceID(const cc_Mesh *mesh, int32_t halfedgeID);
 __host__ __device__  int32_t ccm_HalfedgeEdgeID(const cc_Mesh *mesh, int32_t halfedgeID);
 __host__ __device__ int32_t ccm_HalfedgeVertexID(const cc_Mesh *mesh, int32_t halfedgeID);
 __host__ __device__ int32_t ccm_HalfedgeUvID(const cc_Mesh *mesh, int32_t halfedgeID);
 __host__ __device__  double ccm_HalfedgeSharpness(const cc_Mesh *mesh, int32_t halfedgeID);
 __host__ __device__ cc_VertexPoint ccm_HalfedgeVertexPoint(const cc_Mesh *mesh, int32_t halfedgeID);
 __host__ __device__ cc_VertexUv ccm_HalfedgeVertexUv(const cc_Mesh *mesh, int32_t halfedgeID);
 __host__ __device__  int32_t ccm_CreaseNextID(const cc_Mesh *mesh, int32_t edgeID);
 __host__ __device__  int32_t ccm_CreasePrevID(const cc_Mesh *mesh, int32_t edgeID);
 __host__ __device__  double ccm_CreaseSharpness(const cc_Mesh *mesh, int32_t edgeID);
 __host__ __device__ cc_VertexPoint ccm_VertexPoint(const cc_Mesh *mesh, int32_t vertexID);
 __host__ __device__ cc_VertexUv ccm_Uv(const cc_Mesh *mesh, int32_t uvID);
 __host__ __device__  int32_t ccm_HalfedgeNextID_Quad(int32_t halfedgeID);
 __host__ __device__  int32_t ccm_HalfedgePrevID_Quad(int32_t halfedgeID);
 __host__ __device__  int32_t ccm_HalfedgeFaceID_Quad(int32_t halfedgeID);


/***************
(vertex, edge, face) -> halfedge mappings (O(1))
****************/
 __host__ __device__  int32_t ccm_VertexToHalfedgeID(const cc_Mesh *mesh, int32_t vertexID);
 __host__ __device__  int32_t ccm_EdgeToHalfedgeID(const cc_Mesh *mesh, int32_t edgeID);
 __host__ __device__ int32_t ccm_FaceToHalfedgeID(const cc_Mesh *mesh, int32_t faceID);
 __host__ __device__  int32_t ccm_FaceToHalfedgeID_Quad(int32_t faceID);

// halfedge remappings (O(1))
 __host__ __device__  int32_t ccm_NextVertexHalfedgeID(const cc_Mesh *mesh, int32_t halfedgeID);
 __host__ __device__  int32_t ccm_PrevVertexHalfedgeID(const cc_Mesh *mesh, int32_t halfedgeID);


/*******************************************************************************
 * Vertex to Halfedge Mapping
 *
 * This procedure returns the ID of one of the halfedge that connects a
 * given vertex. This routine has O(depth) complexity.
 *
 */

__host__ __device__  cc_Halfedge_SemiRegular *ccs__Halfedge(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);
__host__ __device__  int32_t ccs__VertexToHalfedgeID_First(const cc_Mesh *cage, int32_t vertexID);
__host__ __device__  cc_Halfedge *ccm__Halfedge(const cc_Mesh *mesh, int32_t halfedgeID);
__host__ __device__  cc_Crease *ccm__Crease(const cc_Mesh *mesh, int32_t edgeID); 
__host__ __device__  int32_t ccm__ScrollFaceHalfedgeID_Quad(int32_t halfedgeID, int32_t direction);
__host__ __device__  const cc_Crease * ccs__Crease(const cc_Subd *subd, int32_t edgeID, int32_t depth);
__host__ __device__  int32_t ccs__EdgeToHalfedgeID_First(const cc_Mesh *cage, int32_t edgeID);
__host__ __device__  uint32_t ccs__HalfedgeVertexUvID(const cc_Subd *subd, int32_t halfedgeID, int32_t depth);