#ifndef CC_LOCAL_SIZE_X
#   define CC_LOCAL_SIZE_X 256
#endif

layout (local_size_x = CC_LOCAL_SIZE_X,
        local_size_y = 1,
        local_size_z = 1) in;

void WriteCrease(int edgeID, in const cc_Crease crease)
{
    ccsu_Creases[edgeID] = crease;
}

void main()
{
    const uint threadID = gl_GlobalInvocationID.x;
    const int edgeCount = ccm_EdgeCount();
    const int edgeID = int(threadID);

    if (edgeID < edgeCount) {
        const int nextID = ccm_CreaseNextID(edgeID);
        const int prevID = ccm_CreasePrevID(edgeID);
        const bool t1 = ccm_CreasePrevID(nextID) == edgeID && nextID != edgeID;
        const bool t2 = ccm_CreaseNextID(prevID) == edgeID && prevID != edgeID;
        const float thisS = 3.0f * ccm_CreaseSharpness(edgeID);
        const float nextS = ccm_CreaseSharpness(nextID);
        const float prevS = ccm_CreaseSharpness(prevID);
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
        WriteCrease(2 * edgeID + 0, newCreases[0]);
        WriteCrease(2 * edgeID + 1, newCreases[1]);
    }
}

