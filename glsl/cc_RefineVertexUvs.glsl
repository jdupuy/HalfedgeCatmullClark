#ifndef CC_LOCAL_SIZE_X
#   define CC_LOCAL_SIZE_X 256
#endif

uniform int u_Depth;

layout (local_size_x = CC_LOCAL_SIZE_X,
        local_size_y = 1,
        local_size_z = 1) in;

void WriteHalfedgeUv(int halfedgeID, vec2 uv, int depth)
{
    const int stride = ccs_CumulativeHalfedgeCountAtDepth(depth);

    ccsu_Halfedges[stride + halfedgeID].uvID = cc__EncodeUv(uv);
}

void main()
{
    const int depth = u_Depth;
    const uint threadID = gl_GlobalInvocationID.x;
    const int halfedgeCount = ccm_HalfedgeCountAtDepth(depth);
    const int halfedgeID = int(threadID);

    if (halfedgeID < halfedgeCount) {
        const int prevID = ccs_HalfedgePrevID(halfedgeID, depth);
        const int nextID = ccs_HalfedgeNextID(halfedgeID, depth);
        const vec2 uv = ccs_HalfedgeVertexUv(halfedgeID, depth);
        const vec2 nextUv = ccs_HalfedgeVertexUv(nextID, depth);
        const vec2 prevUv = ccs_HalfedgeVertexUv(prevID, depth);
        const vec2 edgeUv = (uv + nextUv) * 0.5f;
        const vec2 prevEdgeUv = (uv + prevUv) * 0.5f;
        vec2 faceUv = uv;

        for (int halfedgeIt = ccm_HalfedgeNextID_Quad(halfedgeID);
                 halfedgeIt != halfedgeID;
                 halfedgeIt = ccm_HalfedgeNextID_Quad(halfedgeIt)) {
            const vec2 uv = ccs_HalfedgeVertexUv(halfedgeIt, depth);

            faceUv+= uv;
        }

        faceUv*= 0.25f;

        WriteHalfedgeUv(4 * halfedgeID + 0, uv        , depth);
        WriteHalfedgeUv(4 * halfedgeID + 1, edgeUv    , depth);
        WriteHalfedgeUv(4 * halfedgeID + 2, faceUv    , depth);
        WriteHalfedgeUv(4 * halfedgeID + 3, prevEdgeUv, depth);
    }
}

