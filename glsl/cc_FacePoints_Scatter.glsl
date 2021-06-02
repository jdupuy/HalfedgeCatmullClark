#ifndef CC_LOCAL_SIZE_X
#   define CC_LOCAL_SIZE_X 256
#endif

uniform int u_Depth;

layout (local_size_x = CC_LOCAL_SIZE_X,
        local_size_y = 1,
        local_size_z = 1) in;

void WriteVertex(int vertexID, in const vec3 vertexPoint, int depth)
{
    const int stride = ccs_CumulativeVertexCountAtDepth(depth);
    const int tmp = stride + vertexID;

#define vertexPoints ccsu_VertexPoints
    atomicAdd(vertexPoints[3 * tmp + 0], vertexPoint.x);
    atomicAdd(vertexPoints[3 * tmp + 1], vertexPoint.y);
    atomicAdd(vertexPoints[3 * tmp + 2], vertexPoint.z);
#undef vertexPoints
}

void main()
{
    const int depth = u_Depth;
    const uint threadID = gl_GlobalInvocationID.x;
    const int halfedgeCount = ccm_HalfedgeCountAtDepth(depth);
    const int halfedgeID = int(threadID);

    if (halfedgeID < halfedgeCount) {
        const int vertexCount = ccm_VertexCountAtDepth_Fast(depth);
        const int faceID = ccm_HalfedgeFaceID_Quad(halfedgeID);
        const vec3 vertexPoint = ccs_HalfedgeVertexPoint(halfedgeID, depth);
        const vec3 facePoint = vertexPoint * 0.25f;

        WriteVertex(vertexCount + faceID, facePoint, depth);
    }
}

