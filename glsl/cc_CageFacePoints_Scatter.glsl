#ifndef CC_LOCAL_SIZE_X
#   define CC_LOCAL_SIZE_X 256
#endif

layout (local_size_x = CC_LOCAL_SIZE_X,
        local_size_y = 1,
        local_size_z = 1) in;

void WriteVertex(int vertexID, in const vec3 vertexPoint)
{
#define vertexPoints ccsu_VertexPoints
    atomicAdd(vertexPoints[3 * vertexID + 0], vertexPoint.x);
    atomicAdd(vertexPoints[3 * vertexID + 1], vertexPoint.y);
    atomicAdd(vertexPoints[3 * vertexID + 2], vertexPoint.z);
#undef vertexPoints
}

void main()
{
    const uint threadID = gl_GlobalInvocationID.x;
    const int halfedgeCount = ccm_HalfedgeCount();
    const int halfedgeID = int(threadID);

    if (halfedgeID < halfedgeCount) {
        const int vertexCount = ccm_VertexCount();
        const int faceID = ccm_HalfedgeFaceID(halfedgeID);
        const vec3 vertexPoint = ccm_HalfedgeVertexPoint(halfedgeID);
        int halfedgeIt = ccm_HalfedgeNextID(halfedgeID);
        float n = 1.0f;

        while (halfedgeIt != halfedgeID) {
            halfedgeIt = ccm_HalfedgeNextID(halfedgeIt);
            ++n;
        }

        WriteVertex(vertexCount + faceID, vertexPoint / n);
    }
}

