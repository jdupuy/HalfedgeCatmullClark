#include "Mesh.h"

// (re-)compute catmull clark subdivision
 void ccs_Refine_Gather(cc_Subd *subd);
 void ccs_Refine_Scatter(cc_Subd *subd);
 void ccs_RefineVertexPoints_Gather(cc_Subd *subd);
 void ccs_RefineVertexPoints_Scatter(cc_Subd *subd);
 void ccs_RefineHalfedges(cc_Subd *subd);
 void ccs_RefineCreases(cc_Subd *subd);
#ifndef CC_DISABLE_UV
 void ccs_RefineVertexUvs(cc_Subd *subd);
#endif

// (re-)compute catmull clark vertex points without semi-sharp creases
 void ccs_Refine_NoCreases_Gather(cc_Subd *subd);
 void ccs_Refine_NoCreases_Scatter(cc_Subd *subd);
 void ccs_RefineVertexPoints_NoCreases_Gather(cc_Subd *subd);
 void ccs_RefineVertexPoints_NoCreases_Scatter(cc_Subd *subd);

#ifndef CC_LOG
#    include <stdio.h>
#    define CC_LOG(format, ...) do { fprintf(stdout, format "\n", ##__VA_ARGS__); fflush(stdout); } while(0)
#endif

#ifndef CC_MEMCPY
#    include <string.h>
#    define CC_MEMCPY(dest, src, count) memcpy(dest, src, count)
#endif

#ifndef CC_MEMSET
#    include <string.h>
#    define CC_MEMSET(ptr, value, num) memset(ptr, value, num)
#endif