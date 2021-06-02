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
    const int cageID = 0;
    const int depth = u_Depth;
    const uint threadID = gl_GlobalInvocationID.x;
    const int halfedgeCount = ccm_HalfedgeCountAtDepth(depth);
    const int halfedgeID = int(threadID);
    const int vertexCount = ccm_VertexCountAtDepth_Fast(depth);
    const int faceCount = ccm_FaceCountAtDepth_Fast(depth);

    if (halfedgeID < halfedgeCount) {
        const int twinID = ccs_HalfedgeTwinID(halfedgeID, depth);
        const int edgeID = ccs_HalfedgeEdgeID(halfedgeID, depth);
        const int faceID = ccs_HalfedgeFaceID(halfedgeID, depth);
        const int nextID = ccs_HalfedgeNextID(halfedgeID, depth);
        const float sharp = ccs_CreaseSharpness(edgeID, depth);
        const float edgeWeight = clamp(sharp, 0.0f, 1.0f);
        const vec3 newFacePoint = ccs_VertexPoint(vertexCount + faceID, depth + 1);
        const vec3 oldEdgePoints[2] = {
            ccs_HalfedgeVertexPoint(halfedgeID, depth),
            ccs_HalfedgeVertexPoint(    nextID, depth)
        };
        const vec3 sharpPoint = mix(oldEdgePoints[0],
                                    oldEdgePoints[1],
                                    0.5f) * (twinID < 0 ? 1.0f : 0.5f);
        const vec3 smoothPoint = mix(oldEdgePoints[0], newFacePoint, 0.5f) * 0.5f;
        const vec3 atomicWeight = mix(smoothPoint, sharpPoint, edgeWeight);

        WriteVertex(vertexCount + faceCount + edgeID, atomicWeight, depth);
    }
}

