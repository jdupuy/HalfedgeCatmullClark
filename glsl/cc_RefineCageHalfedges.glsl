#ifndef CC_LOCAL_SIZE_X
#   define CC_LOCAL_SIZE_X 256
#endif

layout (local_size_x = CC_LOCAL_SIZE_X,
        local_size_y = 1,
        local_size_z = 1) in;

void
WriteHalfedge(int halfedgeID, in const cc_Halfedge_SemiRegular halfedge)
{
    ccsu_Halfedges[halfedgeID] = halfedge;
}

void main()
{
    const uint threadID = gl_GlobalInvocationID.x;
    const int halfedgeCount = ccm_HalfedgeCount();
    const int halfedgeID = int(threadID);

    if (halfedgeID < halfedgeCount) {
        const int vertexCount = ccm_VertexCount();
        const int edgeCount = ccm_EdgeCount();
        const int faceCount = ccm_FaceCount();
        const int vertexID = ccm_HalfedgeVertexID(halfedgeID);
        const int twinID = ccm_HalfedgeTwinID(halfedgeID);
        const int prevID = ccm_HalfedgePrevID(halfedgeID);
        const int nextID = ccm_HalfedgeNextID(halfedgeID);
        const int faceID = ccm_HalfedgeFaceID(halfedgeID);
        const int edgeID = ccm_HalfedgeEdgeID(halfedgeID);
        const int prevEdgeID = ccm_HalfedgeEdgeID(prevID);
        const int prevTwinID = ccm_HalfedgeTwinID(prevID);
        const int twinNextID =
            twinID >= 0 ? ccm_HalfedgeNextID(twinID) : -1;
        cc_Halfedge_SemiRegular newHalfedges[4];

        // twinIDs
        newHalfedges[0].twinID = 4 * twinNextID + 3;
        newHalfedges[1].twinID = 4 * nextID     + 2;
        newHalfedges[2].twinID = 4 * prevID     + 1;
        newHalfedges[3].twinID = 4 * prevTwinID + 0;

        // edgeIDs
        newHalfedges[0].edgeID = 2 * edgeID + (halfedgeID > twinID ? 0 : 1);
        newHalfedges[1].edgeID = 2 * edgeCount + halfedgeID;
        newHalfedges[2].edgeID = 2 * edgeCount + prevID;
        newHalfedges[3].edgeID = 2 * prevEdgeID + (prevID > prevTwinID ? 1 : 0);

        // vertexIDs
        newHalfedges[0].vertexID = vertexID;
        newHalfedges[1].vertexID = vertexCount + faceCount + edgeID;
        newHalfedges[2].vertexID = vertexCount + faceID;
        newHalfedges[3].vertexID = vertexCount + faceCount + prevEdgeID;

        // write data
        WriteHalfedge(4 * halfedgeID + 0, newHalfedges[0]);
        WriteHalfedge(4 * halfedgeID + 1, newHalfedges[1]);
        WriteHalfedge(4 * halfedgeID + 2, newHalfedges[2]);
        WriteHalfedge(4 * halfedgeID + 3, newHalfedges[3]);
    }
}

