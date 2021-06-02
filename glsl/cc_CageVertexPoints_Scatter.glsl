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
        const int vertexID = ccm_HalfedgeVertexID(halfedgeID);
        const int edgeID = ccm_HalfedgeEdgeID(halfedgeID);
        const int faceID = ccm_HalfedgeFaceID(halfedgeID);
        const int prevID = ccm_HalfedgePrevID(halfedgeID);
        const int prevEdgeID = ccm_HalfedgeEdgeID(prevID);
        const float thisS = ccm_HalfedgeSharpness(halfedgeID);
        const float prevS = ccm_HalfedgeSharpness(prevID);
        const float creaseWeight = sign(thisS);
        const float prevCreaseWeight = sign(prevS);
        const vec3 newPrevEdgePoint = ccs_VertexPoint(vertexCount + faceCount + prevEdgeID, 1);
        const vec3 newEdgePoint = ccs_VertexPoint(vertexCount + faceCount + edgeID, 1);
        const vec3 newFacePoint = ccs_VertexPoint(vertexCount + faceID, 1);
        const vec3 oldVertexPoint = ccm_VertexPoint(vertexID);
        vec3 cornerPoint = vec3(0.0f);
        vec3 smoothPoint = vec3(0.0f);
        vec3 creasePoint = vec3(0.0f);
        vec3 atomicWeight = vec3(0.0f);
        float avgS = prevS;
        float creaseCount = prevCreaseWeight;
        float valence = 1.0f;
        int forwardIterator, backwardIterator;

        for (forwardIterator = ccm_HalfedgeTwinID(prevID);
             forwardIterator >= 0 && forwardIterator != halfedgeID;
             forwardIterator = ccm_HalfedgeTwinID(forwardIterator)) {
            const int prevID = ccm_HalfedgePrevID(forwardIterator);
            const float prevS = ccm_HalfedgeSharpness(prevID);
            const float prevCreaseWeight = sign(prevS);

            // valence computation
            ++valence;

            // crease computation
            avgS+= prevS;
            creaseCount+= prevCreaseWeight;

            // next vertex halfedge
            forwardIterator = prevID;
        }

        for (backwardIterator = ccm_HalfedgeTwinID(halfedgeID);
             forwardIterator < 0 && backwardIterator >= 0 && backwardIterator != halfedgeID;
             backwardIterator = ccm_HalfedgeTwinID(backwardIterator)) {
            const int nextID = ccm_HalfedgeNextID(backwardIterator);
            const float nextS = ccm_HalfedgeSharpness(nextID);
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

        WriteVertex(vertexID, atomicWeight);
    }
}

