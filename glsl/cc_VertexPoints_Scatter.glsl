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
    const int faceCount = ccm_FaceCountAtDepth_Fast(depth);
    const int vertexCount = ccm_VertexCountAtDepth_Fast(depth);

    if (halfedgeID < halfedgeCount) {
        const int vertexID = ccs_HalfedgeVertexID(halfedgeID, depth);
        const int edgeID = ccs_HalfedgeEdgeID(halfedgeID, depth);
        const int faceID = ccs_HalfedgeFaceID(halfedgeID, depth);
        const int prevID = ccs_HalfedgePrevID(halfedgeID, depth);
        const int prevEdgeID = ccs_HalfedgeEdgeID(prevID, depth);
        const float thisS = ccs_HalfedgeSharpness(halfedgeID, depth);
        const float prevS = ccs_HalfedgeSharpness(    prevID, depth);
        const float creaseWeight = sign(thisS);
        const float prevCreaseWeight = sign(prevS);
        const vec3 newPrevEdgePoint = ccs_VertexPoint(vertexCount + faceCount + prevEdgeID, depth + 1);
        const vec3 newEdgePoint = ccs_VertexPoint(vertexCount + faceCount + edgeID, depth + 1);
        const vec3 newFacePoint = ccs_VertexPoint(vertexCount + faceID, depth + 1);
        const vec3 oldVertexPoint = ccs_VertexPoint(vertexID, depth);
        vec3 cornerPoint = vec3(0.0f);
        vec3 smoothPoint = vec3(0.0f);
        vec3 creasePoint = vec3(0.0f);
        vec3 atomicWeight = vec3(0.0f);
        float avgS = prevS;
        float creaseCount = prevCreaseWeight;
        float valence = 1.0f;
        int forwardIterator, backwardIterator;

        for (forwardIterator = ccs_HalfedgeTwinID(prevID, depth);
             forwardIterator >= 0 && forwardIterator != halfedgeID;
             forwardIterator = ccs_HalfedgeTwinID(forwardIterator, depth)) {
            const int prevID = ccs_HalfedgePrevID(forwardIterator, depth);
            const float prevS = ccs_HalfedgeSharpness(prevID, depth);
            const float prevCreaseWeight = sign(prevS);

            // valence computation
            ++valence;

            // crease computation
            avgS+= prevS;
            creaseCount+= prevCreaseWeight;

            // next vertex halfedge
            forwardIterator = prevID;
        }

        for (backwardIterator = ccs_HalfedgeTwinID(halfedgeID, depth);
             forwardIterator < 0 && backwardIterator >= 0 && backwardIterator != halfedgeID;
             backwardIterator = ccs_HalfedgeTwinID(backwardIterator, depth)) {
            const int nextID = ccs_HalfedgeNextID(backwardIterator, depth);
            const float nextS = ccs_HalfedgeSharpness(nextID, depth);
            const float nextCreaseWeight = sign(nextS);

            // valence computation
            ++valence;

            // crease computation
            avgS+= nextS;
            creaseCount+= nextCreaseWeight;

            // next vertex halfedge
            backwardIterator = nextID;
        }

        // corner point
        cornerPoint = oldVertexPoint / valence;

        // crease computation: V / 4
        creasePoint = (oldVertexPoint + newEdgePoint) * 0.25f * creaseWeight;

        // smooth computation: (4E - F + (n - 3) V) / N
        const vec3 E = newEdgePoint;
        const vec3 F = newFacePoint;
        const vec3 V = oldVertexPoint;
        const float N = valence;
        smoothPoint = (4.0f * E - F + (N - 3.0f) * V) / (N * N);

        // boundary corrections
        if (forwardIterator < 0) {
            creaseCount+= creaseWeight;
            ++valence;

            creasePoint+= (oldVertexPoint + newPrevEdgePoint) * 0.25f * prevCreaseWeight;
        }

        // average sharpness
        avgS/= valence;

        // atomicWeight
        if (creaseCount >= 3.0f || valence == 2.0f) {
            atomicWeight = cornerPoint;
        } else if (creaseCount <= 1.0f) {
            atomicWeight = smoothPoint;
        } else {
            atomicWeight = mix(cornerPoint, creasePoint, clamp(avgS, 0.0f, 1.0f));
        }

        WriteVertex(vertexID, atomicWeight, depth);
    }
}



