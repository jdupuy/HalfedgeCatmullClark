#pragma once
#include "Utilities.h"
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
void ccs__RefineTopology(cc_Subd *subd);

// (re-)compute catmull clark vertex points without semi-sharp creases
 void ccs_Refine_NoCreases_Gather(cc_Subd *subd);
 void ccs_Refine_NoCreases_Scatter(cc_Subd *subd);
 void ccs_RefineVertexPoints_NoCreases_Gather(cc_Subd *subd);
 void ccs_RefineVertexPoints_NoCreases_Scatter(cc_Subd *subd);

#ifndef CC_MEMCPY
#    include <string.h>
#    define CC_MEMCPY(dest, src, count) memcpy(dest, src, count)
#endif

// #ifndef CC_MEMSET
// #    include <string.h>
// #    define CC_MEMSET(ptr, value, num) cudaMemset(ptr, value, num)
// #endif

#ifndef CC_MEMSET
#    include <string.h>
#    define CC_MEMSET(ptr, value, num) memset(ptr, value, num)
#endif

#ifndef CC_PARALLEL_FOR
#   define CC_PARALLEL_FOR    _Pragma("omp parallel for")
#endif
#ifndef CC_BARRIER
#   define CC_BARRIER         _Pragma("omp barrier")
#endif

#ifndef CC_ATOMIC
#   define CC_ATOMIC          _Pragma("omp atomic" )
#endif