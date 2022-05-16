struct cc_Halfedge {
    int twinID;
    int nextID;
    int prevID;
    int faceID;
    int edgeID;
    int vertexID;
    int uvID;
};

struct cc_Halfedge_SemiRegular {
    int twinID;
    int edgeID;
    int vertexID;
#ifndef CC_DISABLE_UV
    int uvID;
#endif
};

struct cc_Crease {
    int nextID;
    int prevID;
    float sharpness;
};


// -----------------------------------------------------------------------------
// Buffers
#ifndef CC_BUFFER_BINDING_CAGE_VERTEX_TO_HALFEDGE
#   error Unspecified Buffer Binding
#endif
#ifndef CC_BUFFER_BINDING_CAGE_EDGE_TO_HALFEDGE
#   error Unspecified Buffer Binding
#endif
#ifndef CC_BUFFER_BINDING_CAGE_FACE_TO_HALFEDGE
#   error Unspecified Buffer Binding
#endif
#ifndef CC_BUFFER_BINDING_CAGE_HALFEDGE
#   error User must specify the binding of the cage halfedge buffer
#endif
#ifndef CC_BUFFER_BINDING_CAGE_CREASE
#   error User must specify the binding of the cage crease buffer
#endif
#ifndef CC_BUFFER_BINDING_CAGE_VERTEX_POINT
#   error User must specify the binding of the cage vertex buffer
#endif
#ifndef CC_BUFFER_BINDING_CAGE_UV
#   error User must specify the binding of the cage uv buffer
#endif
#ifndef CC_BUFFER_BINDING_CAGE_COUNTERS
#   error User must specify the binding of the cage counter
#endif
#ifndef CC_BUFFER_BINDING_SUBD_MAXDEPTH
#   error User must specify the binding of the subd maxDepth buffer
#endif
#ifndef CC_BUFFER_BINDING_SUBD_HALFEDGE
#   error User must specify the binding of the subd halfedge buffer
#endif
#ifndef CC_BUFFER_BINDING_SUBD_VERTEX_POINT
#   error User must specify the binding of the subd vertex buffer
#endif
#ifndef CC_BUFFER_BINDING_SUBD_CREASE
#   error User must specify the binding of the subd crease buffer
#endif

layout(std430, binding = CC_BUFFER_BINDING_CAGE_VERTEX_TO_HALFEDGE)
readonly buffer ccm_HalfedgeToVertexBuffer {
    int ccmu_VertexToHalfedgeIDs[];
};

layout(std430, binding = CC_BUFFER_BINDING_CAGE_EDGE_TO_HALFEDGE)
readonly buffer ccm_EdgeToHalfedgeBuffer {
    int ccmu_EdgeToHalfedgeIDs[];
};

layout(std430, binding = CC_BUFFER_BINDING_CAGE_FACE_TO_HALFEDGE)
readonly buffer ccm_FaceToHalfedgeBuffer {
    int ccmu_FaceToHalfedgeIDs[];
};

layout(std430, binding = CC_BUFFER_BINDING_CAGE_HALFEDGE)
readonly buffer ccm_HalfedgeBuffer {
    cc_Halfedge ccmu_Halfedges[];
};

layout(std430, binding = CC_BUFFER_BINDING_CAGE_CREASE)
readonly buffer ccm_CreaseBuffer {
    cc_Crease ccmu_Creases[];
};

layout(std430, binding = CC_BUFFER_BINDING_CAGE_VERTEX_POINT)
readonly buffer ccm_VertexPointBuffer {
    float ccmu_VertexPoints[];
};

layout(std430, binding = CC_BUFFER_BINDING_CAGE_UV)
readonly buffer ccm_UvBuffer {
    float ccmu_Uvs[];
};

layout(std430, binding = CC_BUFFER_BINDING_CAGE_COUNTERS)
readonly buffer ccm_CounterBuffer {
    int ccmu_VertexCount;
    int ccmu_HalfedgeCount;
    int ccmu_EdgeCount;
    int ccmu_FaceCount;
    int ccmu_UvCount;
};

layout(std430, binding = CC_BUFFER_BINDING_SUBD_MAXDEPTH)
readonly buffer ccs_MaxDepthBuffer {
    int ccsu_MaxDepth;
};

layout(std430, binding = CC_BUFFER_BINDING_SUBD_VERTEX_POINT)
#ifndef CCS_VERTEX_WRITE
readonly
#endif
buffer ccs_VertexPointBuffer {
    float ccsu_VertexPoints[];
};

layout(std430, binding = CC_BUFFER_BINDING_SUBD_HALFEDGE)
#ifndef CCS_HALFEDGE_WRITE
readonly
#endif
buffer ccs_HalfedgeBuffer {
    cc_Halfedge_SemiRegular ccsu_Halfedges[];
};

layout(std430, binding = CC_BUFFER_BINDING_SUBD_CREASE)
#ifndef CCS_CREASE_WRITE
readonly
#endif
buffer ccs_CreaseBuffer {
    cc_Crease ccsu_Creases[];
};


// -----------------------------------------------------------------------------

// mesh queries
int ccm_FaceCount();
int ccm_EdgeCount();
int ccm_HalfedgeCount();
int ccm_CreaseCount();
int ccm_VertexCount();
int ccm_UvCount();

// counts at a given Catmull-Clark subdivision depth
int ccm_HalfedgeCountAtDepth(int depth);
int ccm_FaceCountAtDepth     (int depth);
int ccm_FaceCountAtDepth_Fast(int depth);
int ccm_EdgeCountAtDepth     (int depth);
int ccm_EdgeCountAtDepth_Fast(int depth);
int ccm_VertexCountAtDepth     (int depth);
int ccm_VertexCountAtDepth_Fast(int depth);

// data-access (O(1))
int ccm_HalfedgeTwinID(int halfedgeID);
int ccm_HalfedgePrevID(int halfedgeID);
int ccm_HalfedgeNextID(int halfedgeID);
int ccm_HalfedgeFaceID(int halfedgeID);
int ccm_HalfedgeEdgeID(int halfedgeID);
int ccm_HalfedgeVertexID(int halfedgeID);
int ccm_HalfedgeUvID(int halfedgeID);
float ccm_HalfedgeSharpnnes(int halfedgeID);
vec3 ccm_HalfedgeVertexPoint(int halfedgeID);
vec2 ccm_HalfedgeVertexUv(int halfedgeID);
int ccm_CreaseNextID(int edgeID);
int ccm_CreasePrevID(int edgeID);
float ccm_CreaseSharpness(int edgeID);
vec3 ccm_VertexPoint(int vertexID);
vec2 ccm_Uv(int uvID);
int ccm_HalfedgeNextID_Quad(int halfedgeID);
int ccm_HalfedgePrevID_Quad(int halfedgeID);
int ccm_HalfedgeFaceID_Quad(int halfedgeID);

// (vertex, edge, face) -> halfedge mappings (O(1))
int ccm_VertexToHalfedgeID(int vertexID);
int ccm_EdgeToHalfedgeID(int edgeID);
int ccm_FaceToHalfedgeID(int faceID);
int ccm_FaceToHalfedgeID_Quad(int faceID);

// halfedge remappings (O(1))
int ccm_NextVertexHalfedgeID(int halfedgeID);
int ccm_PrevVertexHalfedgeID(int halfedgeID);

// subd queries
int ccs_MaxDepth();
int ccs_CumulativeFaceCount();
int ccs_CumulativeEdgeCount();
int ccs_CumulativeCreaseCount();
int ccs_CumulativeVertexCount();
int ccs_CumulativeHalfedgeCount();
int ccs_CumulativeFaceCountAtDepth(int depth);
int ccs_CumulativeEdgeCountAtDepth(int depth);
int ccs_CumulativeCreaseCountAtDepth(int depth);
int ccs_CumulativeVertexCountAtDepth(int depth);
int ccs_CumulativeHalfedgeCountAtDepth(int depth);

// O(1) data-access
int ccs_HalfedgeTwinID(int halfedgeID, int depth);
int ccs_HalfedgeNextID(int halfedgeID, int depth);
int ccs_HalfedgePrevID(int halfedgeID, int depth);
int ccs_HalfedgeFaceID(int halfedgeID, int depth);
int ccs_HalfedgeEdgeID(int halfedgeID, int depth);
int ccs_HalfedgeVertexID(int halfedgeID, int depth);
vec3 ccs_HalfedgeVertexPoint(int halfedgeID, int depth);
#ifndef CC_DISABLE_UV
vec2 ccs_HalfedgeVertexUv(int halfedgeID, int depth);
#endif
int ccs_CreaseNextID_Fast(int edgeID, int depth);
int ccs_CreaseNextID     (int edgeID, int depth);
int ccs_CreasePrevID_Fast(int edgeID, int depth);
int ccs_CreasePrevID     (int edgeID, int depth);
float ccs_CreaseSharpness_Fast(int edgeID, int depth);
float ccs_CreaseSharpness     (int edgeID, int depth);
float ccs_HalfedgeSharpness(int halfedgeID, int depth);
vec3 ccs_VertexPoint(int vertexID, int depth);

// halfedge remapping
int ccs_NextVertexHalfedgeID(int halfedgeID, int depth);
int ccs_PrevVertexHalfedgeID(int halfedgeID, int depth);

// (vertex, edge) -> halfedge mappings
int ccs_VertexToHalfedgeID(int vertexID, int depth);
int ccs_EdgeToHalfedgeID(int edgeID, int depth);
int ccs_FaceToHalfedgeID(int edgeID, int depth);

// halfedge normal
vec3 ccs_HalfedgeNormal_Fast(int halfedgeID);
vec3 ccs_HalfedgeNormal(int halfedgeID);



// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------


/*******************************************************************************
 * UV Encoding / Decoding routines
 *
 */
vec2 cc__DecodeUv(int uvEncoded)
{
    const uint tmp = uint(uvEncoded);
    const vec2 uv = vec2(
        ((tmp >>  0) & 0xFFFFu) / 65535.0f,
        ((tmp >> 16) & 0xFFFFu) / 65535.0f
    );

    return uv;
}

int cc__EncodeUv(vec2 uv)
{
    const uint u = uint(round(uv[0] * 65535.0f));
    const uint v = uint(round(uv[1] * 65535.0f));
    const uint tmp = ((u & 0xFFFFu) | ((v & 0xFFFFu) << 16));

    return int(tmp);
}


/*******************************************************************************
 * FaceCount -- Returns the number of faces
 *
 */
int ccm_FaceCount()
{
    return ccmu_FaceCount;
}


/*******************************************************************************
 * EdgeCount -- Returns the number of edges
 *
 */
int ccm_EdgeCount()
{
    return ccmu_EdgeCount;
}


/*******************************************************************************
 * HalfedgeCount -- Returns the number of half edges
 *
 */
int ccm_HalfedgeCount()
{
    return ccmu_HalfedgeCount;
}


/*******************************************************************************
 * CreaseCount -- Returns the number of creases
 *
 */
int ccm_CreaseCount()
{
    return ccm_EdgeCount();
}


/*******************************************************************************
 * VertexCount -- Returns the number of vertices
 *
 */
int ccm_VertexCount()
{
    return ccmu_VertexCount;
}


/*******************************************************************************
 * UvCount -- Returns the number of uvs
 *
 */
int ccm_UvCount()
{
    return ccmu_UvCount;
}


/*******************************************************************************
 * FaceCountAtDepth -- Returns the number of faces at a given subdivision depth
 *
 * The number of faces follows the rule
 *          F^{d+1} = H^d
 * Therefore, the number of half edges at a given subdivision depth d>= 0 is
 *          F^d = 4^{d - 1} H^0,
 * where H0 denotes the number of half-edges of the control cage.
 *
 */
int ccm_FaceCountAtDepth_Fast(int depth)
{
    const int H0 = ccm_HalfedgeCount();

    return (H0 << (2 * (depth - 1)));
}

int ccm_FaceCountAtDepth(int depth)
{
    if (depth == 0) {
        return ccm_FaceCount();
    } else {
        return ccm_FaceCountAtDepth_Fast(depth);
    }
}


/*******************************************************************************
 * EdgeCountAtDepth -- Returns the number of edges at a given subdivision depth
 *
 * The number of edges follows the rule
 *          E^{d+1} = 2 E^d + H^d
 * Therefore, the number of edges at a given subdivision depth d>= 0 is
 *          E^d = 2^{d - 1} (2 E^0 + (2^d - 1) H^0),
 * where H0 and E0 respectively denote the number of half-edges and edges
 * of the control cage.
 *
 */
int ccm_EdgeCountAtDepth_Fast(int depth)
{
    const int E0 = ccm_EdgeCount();
    const int H0 = ccm_HalfedgeCount();
    const int tmp = ~(0xFFFFFFFF << depth); // (2^d - 1)

    return ((E0 << 1) + (tmp * H0)) << (depth - 1);
}

int ccm_EdgeCountAtDepth(int depth)
{
    if (depth == 0) {
        return ccm_EdgeCount();
    } else {
        return ccm_EdgeCountAtDepth_Fast(depth);
    }
}


/*******************************************************************************
 * HalfedgeCountAtDepth -- Returns the number of half edges at a given subd depth
 *
 * The number of half edges is multiplied by 4 at each subdivision step.
 * Therefore, the number of half edges at a given subdivision depth d>= 0 is
 *          4^d H0,
 * where H0 denotes the number of half-edges of the control cage.
 *
 */
int ccm_HalfedgeCountAtDepth(int depth)
{
    const int H0 = ccm_HalfedgeCount();

    return H0 << (depth << 1);
}


/*******************************************************************************
 * CreaseCountAtDepth -- Returns the number of creases at a given subd depth
 *
 * The number of creases is multiplied by 2 at each subdivision step.
 * Therefore, the number of halfedges at a given subdivision depth d>= 0 is
 *          2^d C0,
 * where C0 denotes the number of creases of the control cage.
 *
 */
int ccm_CreaseCountAtDepth(int depth)
{
    const int C0 = ccm_CreaseCount();

    return C0 << depth;
}


/*******************************************************************************
 * VertexCountAtDepth -- Returns the number of vertices at a given subd depth
 *
 * The number of vertices follows the rule
 *          V^{d+1} = V^d + E^d + F^d
 * For a quad mesh, the number of vertices at a given subdivision depth d>= 0 is
 *          V^d = V0 + (2^{d-1} - 1)E0 + (2^{d-1} - 1)^2F0,
 * where:
 * - V0 denotes the number of vertices of the control cage
 * - E0 denotes the number of edges of the control cage
 * - F0 denotes the number of faces of the control cage
 * Note that since the input mesh may contain non-quad faces, we compute
 * the first subdivision step by hand and then apply the formula.
 *
 */
int ccm_VertexCountAtDepth_Fast(int depth)
{
    const int V0 = ccm_VertexCount();
    const int F0 = ccm_FaceCount();
    const int E0 = ccm_EdgeCount();
    const int H0 = ccm_HalfedgeCount();
    const int F1 = H0;
    const int E1 = 2 * E0 + H0;
    const int V1 = V0 + E0 + F0;
    const int tmp =  ~(0xFFFFFFFF << (depth - 1)); // 2^{d-1} - 1

    return V1 + tmp * (E1 + tmp * F1);
}

int ccm_VertexCountAtDepth(int depth)
{
    if (depth == 0) {
        return ccm_VertexCount();
    } else {
        return ccm_VertexCountAtDepth_Fast(depth);
    }
}


/*******************************************************************************
 * Halfedge data accessors
 *
 */
cc_Halfedge ccm__Halfedge(int halfedgeID)
{
    return ccmu_Halfedges[halfedgeID];
}

int ccm_HalfedgeTwinID(int halfedgeID)
{
    return ccm__Halfedge(halfedgeID).twinID;
}

int ccm_HalfedgeNextID(int halfedgeID)
{
    return ccm__Halfedge(halfedgeID).nextID;
}

int ccm_HalfedgePrevID(int halfedgeID)
{
    return ccm__Halfedge(halfedgeID).prevID;
}

int ccm_HalfedgeVertexID(int halfedgeID)
{
    return ccm__Halfedge(halfedgeID).vertexID;
}

int ccm_HalfedgeUvID(int halfedgeID)
{
    return ccm__Halfedge(halfedgeID).uvID;
}

int ccm_HalfedgeEdgeID(int halfedgeID)
{
    return ccm__Halfedge(halfedgeID).edgeID;
}

int ccm_HalfedgeFaceID(int halfedgeID)
{
    return ccm__Halfedge(halfedgeID).faceID;
}

float ccm_HalfedgeSharpness(int halfedgeID)
{
    return ccm_CreaseSharpness(ccm_HalfedgeEdgeID(halfedgeID));
}

vec3 ccm_HalfedgeVertexPoint(int halfedgeID)
{
    return ccm_VertexPoint(ccm_HalfedgeVertexID(halfedgeID));
}

vec2 ccm_HalfedgeVertexUv(int halfedgeID)
{
    return ccm_Uv(ccm_HalfedgeUvID(halfedgeID));
}

cc_Crease ccm__Crease(int edgeID)
{
    return ccmu_Creases[edgeID];
}

int ccm_CreaseNextID(int edgeID)
{
    return ccm__Crease(edgeID).nextID;
}

int ccm_CreasePrevID(int edgeID)
{
    return ccm__Crease(edgeID).prevID;
}

float ccm_CreaseSharpness(int edgeID)
{
    return ccm__Crease(edgeID).sharpness;
}

int ccm_HalfedgeFaceID_Quad(int halfedgeID)
{
    return halfedgeID >> 2;
}


/*******************************************************************************
 * Halfedge Iteration (Quad-only special case)
 *
 */
int ccm__ScrollFaceHalfedgeID_Quad(int halfedgeID, int dir)
{
    const int base = 3;
    const int localID = (halfedgeID & base) + dir;

    return (halfedgeID & ~base) | (localID & base);
}

int ccm_HalfedgeNextID_Quad(int halfedgeID)
{
    return ccm__ScrollFaceHalfedgeID_Quad(halfedgeID, +1);
}

int ccm_HalfedgePrevID_Quad(int halfedgeID)
{
    return ccm__ScrollFaceHalfedgeID_Quad(halfedgeID, -1);
}


/*******************************************************************************
 * Vertex queries
 *
 */
vec3 ccm_VertexPoint(int vertexID)
{
#define vertexPoints ccmu_VertexPoints
    const float x = vertexPoints[3 * vertexID + 0];
    const float y = vertexPoints[3 * vertexID + 1];
    const float z = vertexPoints[3 * vertexID + 2];
#undef vertexPoints

    return vec3(x, y, z);
}

vec2 ccm_Uv(int uvID)
{
#define uvs ccmu_Uvs
    const float x = uvs[2 * uvID + 0];
    const float y = uvs[2 * uvID + 1];
#undef uvs

    return vec2(x, y);
}

/*******************************************************************************
 * VertexToHalfedgeID -- Returns a half edge ID that carries a given vertex
 *
 */
int ccm_VertexToHalfedgeID(int vertexID)
{
    return ccmu_VertexToHalfedgeIDs[vertexID];
}


/*******************************************************************************
 * EdgeToHalfedgeID -- Returns a halfedge associated with a given edge
 *
 */
int ccm_EdgeToHalfedgeID(int edgeID)
{
    return ccmu_EdgeToHalfedgeIDs[edgeID];
}


/*******************************************************************************
 * FaceToHalfedgeID -- Returns a halfedge associated with a given face
 *
 */
int ccm_FaceToHalfedgeID(int faceID)
{
    return ccmu_FaceToHalfedgeIDs[faceID];
}

int ccm_FaceToHalfedgeID_Quad(int faceID)
{
    return faceID << 2;
}


/*******************************************************************************
 * Vertex Halfedge Iteration
 *
 */
int ccm_NextVertexHalfedgeID(int halfedgeID)
{
    const int twinID = ccm_HalfedgeTwinID(halfedgeID);

    return twinID >= 0 ? ccm_HalfedgeNextID(twinID) : -1;
}

int ccm_PrevVertexHalfedgeID(int halfedgeID)
{
    const int prevID = ccm_HalfedgePrevID(halfedgeID);

    return ccm_HalfedgeTwinID(prevID);
}


/*******************************************************************************
 * FaceCountAtDepth -- Returns the accumulated number of faces up to a given subdivision depth
 *
 */
int ccs_CumulativeFaceCountAtDepth(int depth)
{
    return ccs_CumulativeHalfedgeCountAtDepth(depth) >> 2;
}

int ccs_CumulativeFaceCount()
{
    return ccs_CumulativeFaceCountAtDepth(ccs_MaxDepth());
}


/*******************************************************************************
 * EdgeCountAtDepth -- Returns the accumulated number of edges up to a given subdivision depth
 *
 */
int ccs_CumulativeEdgeCountAtDepth(int depth)
{
    const int H0 = ccm_HalfedgeCount();
    const int E0 = ccm_EdgeCount();
    const int H1 = H0 << 2;
    const int E1 = (E0 << 1) + H0;
    const int D = depth;
    const int A = ~(0xFFFFFFFF << D); //  2^{d} - 1

    return (A * (6 * E1 + A * H1 - H1)) / 6;
}

int ccs_CumulativeEdgeCount()
{
    return ccs_CumulativeEdgeCountAtDepth(ccs_MaxDepth());
}


/*******************************************************************************
 * HalfedgeCount -- Returns the total number of half edges stored by the subd
 *
 * The number of half edges is multiplied by 4 at each subdivision step.
 * It follows that the number of half-edges is given by the formula
 *    H = H0 x sum_{d=0}^{D} 4^d
 *      = H0 (4^{D+1} - 1) / 3
 * where D denotes the maximum subdivision depth and H0 the number of
 * half edges in the control mesh.
 *
 */
int ccs_CumulativeHalfedgeCountAtDepth(int maxDepth)
{
    const int D = maxDepth;
    const int H0 = ccm_HalfedgeCount();
    const int H1 = H0 << 2;
    const int tmp = ~(0xFFFFFFFF << (D << 1)); // (4^D - 1)

    return (H1 * tmp) / 3;
}

int ccs_CumulativeHalfedgeCount()
{
    return ccs_CumulativeHalfedgeCountAtDepth(ccs_MaxDepth());
}


/*******************************************************************************
 * CreaseCount -- Returns the total number of creases stored by the subd
 *
 * The number of creases is multiplied by 2 at each subdivision step.
 * It follows that the number of half-edges is given by the formula
 *    C = C0 x sum_{d=0}^{D} 2^d
 *      = C0 (2^{D+1} - 1)
 * where D denotes the maximum subdivision depth and C0 the number of
 * creases in the control mesh.
 *
 */
int ccs_CumulativeCreaseCountAtDepth(int maxDepth)
{
    const int D = maxDepth;
    const int C0 = ccm_CreaseCount();
    const int C1 = C0 << 1;
    const int tmp = ~(0xFFFFFFFF << D); // (2^D - 1)

    return (C1 * tmp);
}

int ccs_CumulativeCreaseCount()
{
    return ccs_CumulativeCreaseCountAtDepth(ccs_MaxDepth());
}


/*******************************************************************************
 * VertexCount -- Returns the total number of vertices stored by the subd
 *
 * The number of vertices increases according to the following formula at
 * each subdivision step:
 *  Vd+1 = Fd + Ed + Vd
 * It follows that the number of vertices is given by the formula
 *   Vd = d F0 + (2^(d+1) - 1) E0 +
 *      = 4 H0 (4^D - 1) / 3
 * where D denotes the maximum subdivition depth and H0 the number of
 * half edges in the control mesh
 *
 */
int ccs_CumulativeVertexCountAtDepth(int depth)
{
    const int V0 = ccm_VertexCount();
    const int F0 = ccm_FaceCount();
    const int E0 = ccm_EdgeCount();
    const int H0 = ccm_HalfedgeCount();
    const int F1 = H0;
    const int E1 = 2 * E0 + H0;
    const int V1 = V0 + E0 + F0;
    const int D = depth;
    const int A =  ~(0xFFFFFFFF << (D     ));     //  2^{d} - 1
    const int B =  ~(0xFFFFFFFF << (D << 1)) / 3; // (4^{d} - 1) / 3

    return A * (E1 - (F1 << 1)) + B * F1 + D * (F1 - E1 + V1);
}

int ccs_CumulativeVertexCount()
{
    return ccs_CumulativeVertexCountAtDepth(ccs_MaxDepth());
}


/*******************************************************************************
 * Max Depth Query
 *
 */
int ccs_MaxDepth()
{
    return ccsu_MaxDepth;
}


/*******************************************************************************
 * Crease queries
 *
 */
cc_Crease ccs__Crease(int edgeID, int depth)
{
    const int stride = ccs_CumulativeCreaseCountAtDepth(depth - 1);

    return ccsu_Creases[stride + edgeID];
}

int ccs_CreaseNextID_Fast(int edgeID, int depth)
{
    return ccs__Crease(edgeID, depth).nextID;
}

int ccs_CreaseNextID(int edgeID, int depth)
{
    const int creaseCount = ccm_CreaseCountAtDepth(depth);

    if (edgeID < creaseCount) {
        return ccs_CreaseNextID_Fast(edgeID, depth);
    } else {
        return edgeID;
    }
}

int ccs_CreasePrevID_Fast(int edgeID, int depth)
{
    return ccs__Crease(edgeID, depth).prevID;
}

int ccs_CreasePrevID(int edgeID, int depth)
{
    const int creaseCount = ccm_CreaseCountAtDepth(depth);

    if (edgeID < creaseCount) {
        return ccs_CreasePrevID_Fast(edgeID, depth);
    } else {
        return edgeID;
    }
}

float ccs_CreaseSharpness_Fast(int edgeID, int depth)
{
    return ccs__Crease(edgeID, depth).sharpness;
}

float ccs_CreaseSharpness(int edgeID, int depth)
{
    const int creaseCount = ccm_CreaseCountAtDepth(depth);

    if (edgeID < creaseCount) {
        return ccs_CreaseSharpness_Fast(edgeID, depth);
    } else {
        return 0.0f;
    }
}


/*******************************************************************************
 * Halfedge queries
 *
 */
cc_Halfedge_SemiRegular ccs__Halfedge(int halfedgeID, int depth)
{
    const int stride = ccs_CumulativeHalfedgeCountAtDepth(depth - 1);

    return ccsu_Halfedges[stride + halfedgeID];
}

int ccs_HalfedgeTwinID(int halfedgeID, int depth)
{
    return ccs__Halfedge(halfedgeID, depth).twinID;
}

int ccs_HalfedgeNextID(int halfedgeID, int depth)
{
    return ccm_HalfedgeNextID_Quad(halfedgeID);
}

int ccs_HalfedgePrevID(int halfedgeID, int depth)
{
    return ccm_HalfedgePrevID_Quad(halfedgeID);
}

int ccs_HalfedgeFaceID(int halfedgeID, int depth)
{
    return ccm_HalfedgeFaceID_Quad(halfedgeID);
}

int ccs_HalfedgeEdgeID(int halfedgeID, int depth)
{
    return ccs__Halfedge(halfedgeID, depth).edgeID;
}

int ccs_HalfedgeVertexID(int halfedgeID, int depth)
{
    return ccs__Halfedge(halfedgeID, depth).vertexID;
}

float ccs_HalfedgeSharpness(int halfedgeID, int depth)
{
    const int edgeID = ccs_HalfedgeEdgeID(halfedgeID, depth);

    return ccs_CreaseSharpness(edgeID, depth);
}

vec3 ccs_HalfedgeVertexPoint(int halfedgeID, int depth)
{
    const int vertexID = ccs_HalfedgeVertexID(halfedgeID, depth);

    return ccs_VertexPoint(vertexID, depth);
}

#ifndef CC_DISABLE_UV
int ccs__HalfedgeUvID(int halfedgeID, int depth)
{
    return ccs__Halfedge(halfedgeID, depth).uvID;
}

vec2 ccs_HalfedgeVertexUv(int halfedgeID, int depth)
{
    return cc__DecodeUv(ccs__HalfedgeUvID(halfedgeID, depth));
}
#endif


/*******************************************************************************
 * Vertex queries
 *
 */
vec3 ccs_VertexPoint(int vertexID, int depth)
{
    const int stride = ccs_CumulativeVertexCountAtDepth(depth - 1);
    const int tmp = 3 * (stride + vertexID);

#define vertexPoints ccsu_VertexPoints
    const float x = vertexPoints[tmp + 0];
    const float y = vertexPoints[tmp + 1];
    const float z = vertexPoints[tmp + 2];
#undef vertexPoints

    return vec3(x, y, z);
}


/*******************************************************************************
 * Normal computation
 *
 */
vec3 ccs_HalfedgeNormal_Fast(int halfedgeID)
{
    const int maxDepth = ccs_MaxDepth();
    const int nextID = ccm_HalfedgeNextID_Quad(halfedgeID);
    const int prevID = ccm_HalfedgePrevID_Quad(halfedgeID);
    const vec3 v0 = ccs_HalfedgeVertexPoint(halfedgeID, maxDepth);
    const vec3 v1 = ccs_HalfedgeVertexPoint(prevID    , maxDepth);
    const vec3 v2 = ccs_HalfedgeVertexPoint(nextID    , maxDepth);

    return normalize(cross(v2 - v0, v1 - v0));
}

vec3 ccs_HalfedgeNormal(int halfedgeID)
{
    const int maxDepth = ccs_MaxDepth();
    const vec3 halfedgeNormal = ccs_HalfedgeNormal_Fast(halfedgeID);
    vec3 averageNormal = vec3(0.0f);
    int halfedgeIterator;

    for (halfedgeIterator = ccs_PrevVertexHalfedgeID(halfedgeID, maxDepth);
         halfedgeIterator >= 0 && halfedgeIterator != halfedgeID;
         halfedgeIterator = ccs_PrevVertexHalfedgeID(halfedgeIterator, maxDepth)) {
        averageNormal+= ccs_HalfedgeNormal_Fast(halfedgeIterator);
    }

    if (halfedgeIterator < 0)
        return halfedgeNormal;
    else
        return normalize(halfedgeNormal + averageNormal);
}


/*******************************************************************************
 * VertexHalfedge Iteraion
 *
 */
int ccs_PrevVertexHalfedgeID(int halfedgeID, int depth)
{
    const int prevID = ccm_HalfedgePrevID_Quad(halfedgeID);

    return ccs_HalfedgeTwinID(prevID, depth);
}

int ccs_NextVertexHalfedgeID(int halfedgeID, int depth)
{
    const int twinID = ccs_HalfedgeTwinID(halfedgeID, depth);

    return ccm_HalfedgeNextID_Quad(twinID);
}


/*******************************************************************************
 * Edge to Halfedge Mapping
 *
 * This procedure returns one of the ID of one of the half edge that constitutes
 * the edge. This routine has O(depth) complexity.
 *
 */
int ccs__EdgeToHalfedgeID_First(int edgeID)
{
    const int edgeCount = ccm_EdgeCount();

    if /* [2E, 2E + H) */ (edgeID >= 2 * edgeCount) {
        const int halfedgeID = edgeID - 2 * edgeCount;
        const int nextID = ccm_HalfedgeNextID(halfedgeID);

        return max(4 * halfedgeID + 1, 4 * nextID + 2);

    } else if /* */ ((edgeID & 1) == 1) {
        const int halfedgeID = ccm_EdgeToHalfedgeID(edgeID >> 1);
        const int nextID = ccm_HalfedgeNextID(halfedgeID);

        return 4 * nextID + 3;

    } else /* */ {
        const int halfedgeID = ccm_EdgeToHalfedgeID(edgeID >> 1);

        return 4 * halfedgeID + 0;
    }
}

int ccs_EdgeToHalfedgeID(int edgeID, int depth)
{
    uint heap = 1u;
    int edgeHalfedgeID = 0;
    int heapDepth = depth;

    // build heap
    for (; heapDepth > 1; --heapDepth) {
        const int edgeCount = ccm_EdgeCountAtDepth_Fast(heapDepth - 1);

        if /* [2E, 2E + H) */ (edgeID >= 2 * edgeCount) {
            const int halfedgeID = edgeID - 2 * edgeCount;
            const int nextID = ccm_HalfedgeNextID_Quad(halfedgeID);

            edgeHalfedgeID = max(4 * halfedgeID + 1, 4 * nextID + 2);
            break;
        } else {
            heap = (heap << 1) | (edgeID & 1);
            edgeID>>= 1;
        }
    }

    // initialize root cfg
    if (heapDepth == 1) {
        edgeHalfedgeID = ccs__EdgeToHalfedgeID_First(edgeID);
    }

    // read heap
    while (heap > 1u) {
        if ((heap & 1u) == 1u) {
            const int nextID = ccm_HalfedgeNextID_Quad(edgeHalfedgeID);

            edgeHalfedgeID = 4 * nextID + 3;
        } else {
            edgeHalfedgeID = 4 * edgeHalfedgeID + 0;
        }

        heap>>= 1;
    }

    return edgeHalfedgeID;
}


/*******************************************************************************
 * Vertex to Halfedge Mapping
 *
 * This procedure returns the ID of one of the half edge that connects a
 * given vertex. This routine has O(depth) complexity.
 *
 */
int ccs__VertexToHalfedgeID_First(int vertexID)
{
    const int vertexCount = ccm_VertexCount();
    const int faceCount = ccm_FaceCount();

    if /* [V + F, V + F + E) */ (vertexID >= vertexCount + faceCount) {
        const int edgeID = vertexID - vertexCount - faceCount;

        return 4 * ccm_EdgeToHalfedgeID(edgeID) + 1;

    } else if /* [V, V + F) */ (vertexID >= vertexCount) {
        const int faceID = vertexID - vertexCount;

        return 4 * ccm_FaceToHalfedgeID(faceID) + 2;

    } else /* [0, V) */ {

        return 4 * ccm_VertexToHalfedgeID(vertexID) + 0;
    }
}

int ccs_VertexToHalfedgeID(int vertexID, int depth)
{
    int stride = 0;
    int halfedgeID = 0;
    int heapDepth = depth;

    // build heap
    for (; heapDepth > 1; --heapDepth) {
        const int vertexCount = ccm_VertexCountAtDepth_Fast(heapDepth - 1);
        const int faceCount = ccm_FaceCountAtDepth_Fast(heapDepth - 1);

        if /* [V + F, V + F + E) */ (vertexID >= vertexCount + faceCount) {
            const int edgeID = vertexID - faceCount - vertexCount;

            halfedgeID = 4 * ccs_EdgeToHalfedgeID(edgeID, heapDepth - 1) + 1;
            break;
        } else if /* [V, V + F) */ (vertexID >= vertexCount) {
            const int faceID = vertexID - vertexCount;

            halfedgeID = 4 * ccm_FaceToHalfedgeID_Quad(faceID) + 2;
            break;
        } else /* [0, V) */ {
            stride+= 2;
        }
    }

    // initialize root cfg
    if (heapDepth == 1) {
        halfedgeID = ccs__VertexToHalfedgeID_First(vertexID);
    }

    return (halfedgeID << stride);
}
