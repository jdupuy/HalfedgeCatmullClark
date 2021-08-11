#ifndef CC_LOCAL_SIZE_X
#   define CC_LOCAL_SIZE_X 256
#endif

uniform int u_Depth;

layout (local_size_x = CC_LOCAL_SIZE_X,
        local_size_y = 1,
        local_size_z = 1) in;

void WriteCrease(int edgeID, in const cc_Crease crease, int depth)
{
    const int stride = ccs_CumulativeCreaseCountAtDepth(depth);

    ccsu_Creases[stride + edgeID] = crease;
}

void main()
{
    const int depth = u_Depth;
    const uint threadID = gl_GlobalInvocationID.x;
    const int creaseCount = ccm_CreaseCountAtDepth(depth);
    const int edgeID = int(threadID);

    if (edgeID < creaseCount) {
        const int nextID = ccs_CreaseNextID_Fast(edgeID, depth);
        const int prevID = ccs_CreasePrevID_Fast(edgeID, depth);
        const bool t1 = ccs_CreasePrevID_Fast(nextID, depth) == edgeID && nextID != edgeID;
        const bool t2 = ccs_CreaseNextID_Fast(prevID, depth) == edgeID && prevID != edgeID;
        const float thisS = 3.0f * ccs_CreaseSharpness_Fast(edgeID, depth);
        const float nextS = ccs_CreaseSharpness_Fast(nextID, depth);
        const float prevS = ccs_CreaseSharpness_Fast(prevID, depth);
        cc_Crease newCreases[2];

        // next rule
        newCreases[0].nextID = 2 * edgeID + 1;
        newCreases[1].nextID = 2 * nextID + (t1 ? 0 : 1);

        // prev rule
        newCreases[0].prevID = 2 * prevID + (t2 ? 1 : 0);
        newCreases[1].prevID = 2 * edgeID + 0;

        // sharpness rule
        newCreases[0].sharpness = max(0.0f, (prevS + thisS) / 4.0f - 1.0f);
        newCreases[1].sharpness = max(0.0f, (thisS + nextS) / 4.0f - 1.0f);

        // write data
        WriteCrease(2 * edgeID + 0, newCreases[0], depth);
        WriteCrease(2 * edgeID + 1, newCreases[1], depth);
    }
}

