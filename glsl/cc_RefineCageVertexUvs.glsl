#ifndef CC_LOCAL_SIZE_X
#   define CC_LOCAL_SIZE_X 256
#endif

layout (local_size_x = CC_LOCAL_SIZE_X,
        local_size_y = 1,
        local_size_z = 1) in;

void WriteHalfedgeUv(int halfedgeID, vec2 uv)
{
    ccsu_Halfedges[halfedgeID].uvID = cc__EncodeUv(uv);
}

void main()
{
    const uint threadID = gl_GlobalInvocationID.x;
    const int halfedgeCount = ccm_HalfedgeCount();
    const int halfedgeID = int(threadID);

    if (halfedgeID < halfedgeCount) {
        const int prevID = ccm_HalfedgePrevID(halfedgeID);
        const int nextID = ccm_HalfedgeNextID(halfedgeID);
        const vec2 uv = ccm_HalfedgeVertexUv(halfedgeID);
        const vec2 nextUv = ccm_HalfedgeVertexUv(nextID);
        const vec2 prevUv = ccm_HalfedgeVertexUv(prevID);
        const vec2 edgeUv = (uv + nextUv) * 0.5f;
        const vec2 prevEdgeUv = (uv + prevUv) * 0.5f;
        vec2 faceUv = uv;
        float m = 1.0f;

        for (int halfedgeIt = ccm_HalfedgeNextID(halfedgeID);
                 halfedgeIt != halfedgeID;
                 halfedgeIt = ccm_HalfedgeNextID(halfedgeIt)) {
            const vec2 uv = ccm_HalfedgeVertexUv(halfedgeIt);

            faceUv+= uv;
            ++m;
        }

        faceUv/= m;

        WriteHalfedgeUv(4 * halfedgeID + 0, uv);
        WriteHalfedgeUv(4 * halfedgeID + 1, edgeUv);
        WriteHalfedgeUv(4 * halfedgeID + 2, faceUv);
        WriteHalfedgeUv(4 * halfedgeID + 3, prevEdgeUv);
    }
}

