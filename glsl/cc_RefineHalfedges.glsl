#ifndef CC_LOCAL_SIZE_X
#   define CC_LOCAL_SIZE_X 256
#endif

uniform int u_Depth;

layout (local_size_x = CC_LOCAL_SIZE_X,
        local_size_y = 1,
        local_size_z = 1) in;

void
WriteHalfedge(int halfedgeID, in const cc_Halfedge_SemiRegular halfedge, int depth)
{
    const int stride = ccs_CumulativeHalfedgeCountAtDepth(depth);

    ccsu_Halfedges[stride + halfedgeID] = halfedge;
}

void main()
{
    const int cageID = 0;
    const int depth = u_Depth;
    const uint threadID = gl_GlobalInvocationID.x;
    const int halfedgeCount = ccm_HalfedgeCountAtDepth(depth);
    const int halfedgeID = int(threadID);

    if (halfedgeID < halfedgeCount) {
        const int vertexCount = ccm_VertexCountAtDepth_Fast(depth);
        const int edgeCount = ccm_EdgeCountAtDepth_Fast(depth);
        const int faceCount = ccm_FaceCountAtDepth_Fast(depth);
        const int vertexID = ccs_HalfedgeVertexID(halfedgeID, depth);
        const int twinID = ccs_HalfedgeTwinID(halfedgeID, depth);
        const int prevID = ccs_HalfedgePrevID(halfedgeID, depth);
        const int nextID = ccs_HalfedgeNextID(halfedgeID, depth);
        const int faceID = ccs_HalfedgeFaceID(halfedgeID, depth);
        const int edgeID = ccs_HalfedgeEdgeID(halfedgeID, depth);
        const int prevEdgeID = ccs_HalfedgeEdgeID(prevID, depth);
        const int prevTwinID = ccs_HalfedgeTwinID(prevID, depth);
        const int twinNextID = ccs_HalfedgeNextID(twinID, depth);
        cc_Halfedge_SemiRegular halfedges[4];

        // twinIDs
        halfedges[0].twinID = 4 * twinNextID + 3;
        halfedges[1].twinID = 4 * nextID     + 2;
        halfedges[2].twinID = 4 * prevID     + 1;
        halfedges[3].twinID = 4 * prevTwinID + 0;

        // edgeIDs
        halfedges[0].edgeID = 2 * edgeID + (halfedgeID > twinID ? 0 : 1);
        halfedges[1].edgeID = 2 * edgeCount + halfedgeID;
        halfedges[2].edgeID = 2 * edgeCount + prevID;
        halfedges[3].edgeID = 2 * prevEdgeID + (prevID > prevTwinID ? 1 : 0);

        // vertexIDs
        halfedges[0].vertexID = vertexID;
        halfedges[1].vertexID = vertexCount + faceCount + edgeID;
        halfedges[2].vertexID = vertexCount + faceID;
        halfedges[3].vertexID = vertexCount + faceCount + prevEdgeID;

        WriteHalfedge(4 * halfedgeID + 0, halfedges[0], depth);
        WriteHalfedge(4 * halfedgeID + 1, halfedges[1], depth);
        WriteHalfedge(4 * halfedgeID + 2, halfedges[2], depth);
        WriteHalfedge(4 * halfedgeID + 3, halfedges[3], depth);
    }
}

