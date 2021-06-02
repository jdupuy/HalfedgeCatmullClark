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
    const int vertexCount = ccm_VertexCount();
    const int faceCount = ccm_FaceCount();

    if (halfedgeID < halfedgeCount) {
        const int faceID = ccm_HalfedgeFaceID(halfedgeID);
        const int edgeID = ccm_HalfedgeEdgeID(halfedgeID);
        const int twinID = ccm_HalfedgeTwinID(halfedgeID);
        const int nextID = ccm_HalfedgeNextID(halfedgeID);
        const float sharp = ccm_CreaseSharpness(edgeID);
        const float edgeWeight = clamp(sharp, 0.0f, 1.0f);
        const vec3 newFacePoint = ccs_VertexPoint(vertexCount + faceID, 1);
        const vec3 oldEdgePoints[2] = vec3[2](
            ccm_HalfedgeVertexPoint(halfedgeID),
            ccm_HalfedgeVertexPoint(nextID)
        );
        const vec3 sharpPoint = mix(oldEdgePoints[0],
                                    oldEdgePoints[1],
                                    0.5f) * (twinID < 0 ? 1.0f : 0.5f);
        const vec3 smoothPoint = mix(oldEdgePoints[0], newFacePoint, 0.5f) * 0.5f;
        const vec3 atomicWeight = mix(smoothPoint, sharpPoint, edgeWeight);

        WriteVertex(vertexCount + faceCount + edgeID, atomicWeight);
    }
}

