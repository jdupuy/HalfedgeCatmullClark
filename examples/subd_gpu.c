
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "glad/glad.h"
#include "GLFW/glfw3.h"

#define LOG(fmt, ...) fprintf(stdout, fmt "\n", ##__VA_ARGS__); fflush(stdout);

#define CC_DISABLE_UV
#define CC_IMPLEMENTATION
#include "CatmullClark.h"

#define DJ_OPENGL_IMPLEMENTATION
#include "dj_opengl.h"

#ifndef PATH_TO_SRC_DIRECTORY
#   define PATH_TO_SRC_DIRECTORY "./"
#endif

struct Window {
    const char *name;
    int32_t width, height;
    struct {
        int32_t major, minor;
    } glversion;
    GLFWwindow* handle;
} g_window = {
    "Catmull Clark",
    720, 720,
    {4, 5},
    NULL
};

enum {
    BUFFER_CAGE_VERTEX_TO_HALFEDGE,
    BUFFER_CAGE_EDGE_TO_HALFEDGE,
    BUFFER_CAGE_FACE_TO_HALFEDGE,
    BUFFER_CAGE_HALFEDGES,
    BUFFER_CAGE_CREASES,
    BUFFER_CAGE_VERTEX_POINTS,
    BUFFER_CAGE_VERTEX_UVS,
    BUFFER_CAGE_COUNTERS,
    BUFFER_SUBD_MAXDEPTH,
    BUFFER_SUBD_HALFEDGES,
    BUFFER_SUBD_CREASES,
    BUFFER_SUBD_VERTEX_POINTS,

    BUFFER_COUNT
};

enum {
    PROGRAM_SUBD_CAGE_HALFEDGES,
    PROGRAM_SUBD_CAGE_CREASES,
    PROGRAM_SUBD_CAGE_FACE_POINTS,
    PROGRAM_SUBD_CAGE_EDGE_POINTS,
    PROGRAM_SUBD_CAGE_VERTEX_POINTS,
    PROGRAM_SUBD_CAGE_VERTEX_UVS,
    PROGRAM_SUBD_HALFEDGES,
    PROGRAM_SUBD_CREASES,
    PROGRAM_SUBD_FACE_POINTS,
    PROGRAM_SUBD_EDGE_POINTS,
    PROGRAM_SUBD_VERTEX_POINTS,
    PROGRAM_SUBD_VERTEX_UVS,

    PROGRAM_COUNT
};

struct OpenGL {
    GLuint programs[PROGRAM_COUNT];
    GLuint buffers[BUFFER_COUNT];
} g_gl = {
    {0},
    {0}
};

struct SubdManager {
    int32_t computeShaderLocalSize;
} g_subd = {
    5
};

#define PATH_TO_SHADER_DIRECTORY PATH_TO_SRC_DIRECTORY "../glsl/"

static void
LoadCatmullClarkLibrary(
    djg_program *djgp,
    bool halfedgeWrite,
    bool creaseWrite,
    bool vertexWrite
) {
    djgp_push_string(djgp, "#extension GL_NV_shader_atomic_float: require\n");
    djgp_push_string(djgp, "#extension GL_NV_shader_thread_shuffle: require\n");
#ifdef CC_DISABLE_UV
    djgp_push_string(djgp, "#define CC_DISABLE_UV\n");
#endif

    if (halfedgeWrite) {
        djgp_push_string(djgp, "#define CCS_HALFEDGE_WRITE\n");
    }
    if (creaseWrite) {
        djgp_push_string(djgp, "#define CCS_CREASE_WRITE\n");
    }
    if (vertexWrite) {
        djgp_push_string(djgp, "#define CCS_VERTEX_WRITE\n");
    }

    djgp_push_string(djgp,
                     "#define CC_LOCAL_SIZE_X %i\n",
                     1 << g_subd.computeShaderLocalSize);
    djgp_push_string(djgp,
                     "#define CC_BUFFER_BINDING_CAGE_VERTEX_TO_HALFEDGE %i\n",
                     BUFFER_CAGE_VERTEX_TO_HALFEDGE);
    djgp_push_string(djgp,
                     "#define CC_BUFFER_BINDING_CAGE_EDGE_TO_HALFEDGE %i\n",
                     BUFFER_CAGE_EDGE_TO_HALFEDGE);
    djgp_push_string(djgp,
                     "#define CC_BUFFER_BINDING_CAGE_FACE_TO_HALFEDGE %i\n",
                     BUFFER_CAGE_FACE_TO_HALFEDGE);
    djgp_push_string(djgp,
                     "#define CC_BUFFER_BINDING_CAGE_HALFEDGE %i\n",
                     BUFFER_CAGE_HALFEDGES);
    djgp_push_string(djgp,
                     "#define CC_BUFFER_BINDING_CAGE_CREASE %i\n",
                     BUFFER_CAGE_CREASES);
    djgp_push_string(djgp,
                     "#define CC_BUFFER_BINDING_CAGE_VERTEX_POINT %i\n",
                     BUFFER_CAGE_VERTEX_POINTS);
    djgp_push_string(djgp,
                     "#define CC_BUFFER_BINDING_CAGE_UV %i\n",
                     BUFFER_CAGE_VERTEX_UVS);
    djgp_push_string(djgp,
                     "#define CC_BUFFER_BINDING_CAGE_COUNTERS %i\n",
                     BUFFER_CAGE_COUNTERS);
    djgp_push_string(djgp,
                     "#define CC_BUFFER_BINDING_SUBD_MAXDEPTH %i\n",
                     BUFFER_SUBD_MAXDEPTH);
    djgp_push_string(djgp,
                     "#define CC_BUFFER_BINDING_SUBD_HALFEDGE %i\n",
                     BUFFER_SUBD_HALFEDGES);
    djgp_push_string(djgp,
                     "#define CC_BUFFER_BINDING_SUBD_CREASE %i\n",
                     BUFFER_SUBD_CREASES);
    djgp_push_string(djgp,
                     "#define CC_BUFFER_BINDING_SUBD_VERTEX_POINT %i\n",
                     BUFFER_SUBD_VERTEX_POINTS);

    djgp_push_file(djgp, PATH_TO_SHADER_DIRECTORY "CatmullClark_Scatter.glsl");
}

static bool
LoadCatmullClarkProgram(
    int32_t programID,
    const char *sourceFile,
    bool halfEdgeWrite,
    bool creaseWrite,
    bool vertexWrite
) {
    djg_program *djgp = djgp_create();
    GLuint *glp = &g_gl.programs[programID];

    LoadCatmullClarkLibrary(djgp, halfEdgeWrite, creaseWrite, vertexWrite);
    djgp_push_file(djgp, sourceFile);
    djgp_push_string(djgp, "#ifdef COMPUTE_SHADER\n#endif");
    if (!djgp_to_gl(djgp, 450, false, true, glp)) {
        djgp_release(djgp);

        return false;
    }

    djgp_release(djgp);

    return glGetError() == GL_NO_ERROR;
}

bool LoadCageFacePointsProgram()
{
    LOG("Loading {Program-Cage-Face-Points}");
    const char *srcFile = PATH_TO_SHADER_DIRECTORY "cc_CageFacePoints_Scatter.glsl";

    return LoadCatmullClarkProgram(PROGRAM_SUBD_CAGE_FACE_POINTS, srcFile, false, false, true);
}

bool LoadCageEdgePointsProgram()
{
    LOG("Loading {Program-Cage-Edge-Points}");
    const char *srcFile = PATH_TO_SHADER_DIRECTORY "cc_CageEdgePoints_Scatter.glsl";

    return LoadCatmullClarkProgram(PROGRAM_SUBD_CAGE_EDGE_POINTS, srcFile, false, false, true);
}

bool LoadCageVertexPointsProgram()
{
    LOG("Loading {Program-Cage-Vertex-Points}");
    const char *srcFile = PATH_TO_SHADER_DIRECTORY "cc_CageVertexPoints_Scatter.glsl";

    return LoadCatmullClarkProgram(PROGRAM_SUBD_CAGE_VERTEX_POINTS, srcFile, false, false, true);
}

bool LoadCageHalfedgeRefinementProgram()
{
    LOG("Loading {Program-Refine-Cage-Halfedges}");
    const char *srcFile = PATH_TO_SHADER_DIRECTORY "cc_RefineCageHalfedges.glsl";

    return LoadCatmullClarkProgram(PROGRAM_SUBD_CAGE_HALFEDGES, srcFile, true, false, false);
}

bool LoadCageCreaseRefinementProgram()
{
    LOG("Loading {Program-Refine-Cage-Creases}");
    const char *srcFile = PATH_TO_SHADER_DIRECTORY "cc_RefineCageCreases.glsl";

    return LoadCatmullClarkProgram(PROGRAM_SUBD_CAGE_CREASES, srcFile, false, true, false);
}

bool LoadFacePointsProgram()
{
    LOG("Loading {Program-Face-Points}");
    const char *srcFile = PATH_TO_SHADER_DIRECTORY "cc_FacePoints_Scatter.glsl";

    return LoadCatmullClarkProgram(PROGRAM_SUBD_FACE_POINTS, srcFile, false, false, true);
}

bool LoadEdgeRefinementProgram()
{
    LOG("Loading {Program-Edge-Points}");
    const char *srcFile = PATH_TO_SHADER_DIRECTORY "cc_EdgePoints_Scatter.glsl";

    return LoadCatmullClarkProgram(PROGRAM_SUBD_EDGE_POINTS, srcFile, false, false, true);
}

bool LoadVertexRefinementProgram()
{
    LOG("Loading {Program-Vertex-Points}");
    const char *srcFile = PATH_TO_SHADER_DIRECTORY "cc_VertexPoints_Scatter.glsl";

    return LoadCatmullClarkProgram(PROGRAM_SUBD_VERTEX_POINTS, srcFile, false, false, true);
}

bool LoadHalfedgeRefinementProgram()
{
    LOG("Loading {Program-Refine-Halfedges}");
    const char *srcFile = PATH_TO_SHADER_DIRECTORY "cc_RefineHalfedges.glsl";

    return LoadCatmullClarkProgram(PROGRAM_SUBD_HALFEDGES, srcFile, true, false, false);
}

bool LoadCreaseRefinementProgram()
{
    LOG("Loading {Program-Refine-Creases}");
    const char *srcFile = PATH_TO_SHADER_DIRECTORY "cc_RefineCreases.glsl";

    return LoadCatmullClarkProgram(PROGRAM_SUBD_CREASES, srcFile, false, true, false);
}

#ifndef CC_DISABLE_UV
bool LoadCageVertexUvRefinementProgram()
{
    LOG("Loading {Program-Refine-Cage-Uvs}");
    const char *srcFile = PATH_TO_SHADER_DIRECTORY "cc_RefineCageVertexUvs.glsl";

    return LoadCatmullClarkProgram(PROGRAM_SUBD_CAGE_VERTEX_UVS, srcFile, true, false, false);
}

bool LoadVertexUvRefinementProgram()
{
    LOG("Loading {Program-Refine-Vertex-Uvs}");
    const char *srcFile = PATH_TO_SHADER_DIRECTORY "cc_RefineVertexUvs.glsl";

    return LoadCatmullClarkProgram(PROGRAM_SUBD_VERTEX_UVS, srcFile, true, false, false);
}
#endif

bool LoadPrograms()
{
    bool success = true;

    if (success) success = LoadCageHalfedgeRefinementProgram();
    if (success) success = LoadHalfedgeRefinementProgram();
    if (success) success = LoadCageCreaseRefinementProgram();
    if (success) success = LoadCreaseRefinementProgram();
    if (success) success = LoadCageFacePointsProgram();
    if (success) success = LoadCageEdgePointsProgram();
    if (success) success = LoadCageVertexPointsProgram();
    if (success) success = LoadFacePointsProgram();
    if (success) success = LoadEdgeRefinementProgram();
    if (success) success = LoadVertexRefinementProgram();
#ifndef CC_DISABLE_UV
    if (success) success = LoadCageVertexUvRefinementProgram();
    if (success) success = LoadVertexUvRefinementProgram();
#endif

    return success;
}
#undef PATH_TO_SHADER_DIRECTORY

GLuint
LoadCatmullClarkBuffer(
    int32_t bufferID,
    GLsizeiptr bufferByteSize,
    const void *data,
    GLbitfield flags
) {
    GLuint *buffer = &g_gl.buffers[bufferID];

    glGenBuffers(1, buffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, *buffer);
    glBufferStorage(GL_SHADER_STORAGE_BUFFER,
                    bufferByteSize,
                    data,
                    flags);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bufferID, *buffer);

    return glGetError() == GL_NO_ERROR;
}

bool LoadCageVertexToHalfedgeIDsBuffer(const cc_Mesh *cage)
{
    return LoadCatmullClarkBuffer(BUFFER_CAGE_VERTEX_TO_HALFEDGE,
                                  ccm_VertexCount(cage) * sizeof(int32_t),
                                  cage->vertexToHalfedgeIDs,
                                  0);
}

bool LoadCageEdgeToHalfedgeIDsBuffer(const cc_Mesh *cage)
{
    return LoadCatmullClarkBuffer(BUFFER_CAGE_EDGE_TO_HALFEDGE,
                                  ccm_EdgeCount(cage) * sizeof(int32_t),
                                  cage->edgeToHalfedgeIDs,
                                  0);
}

bool LoadCageFaceToHalfedgeIDsBuffer(const cc_Mesh *cage)
{
    return LoadCatmullClarkBuffer(BUFFER_CAGE_FACE_TO_HALFEDGE,
                                  ccm_FaceCount(cage) * sizeof(int32_t),
                                  cage->faceToHalfedgeIDs,
                                  0);
}

bool LoadCageHalfedgeBuffer(const cc_Mesh *cage)
{
    return LoadCatmullClarkBuffer(BUFFER_CAGE_HALFEDGES,
                                  sizeof(cc_Halfedge) * ccm_HalfedgeCount(cage),
                                  cage->halfedges,
                                  0);
}

bool LoadCageCreaseBuffer(const cc_Mesh *cage)
{
    return LoadCatmullClarkBuffer(BUFFER_CAGE_CREASES,
                                  sizeof(cc_Crease) * ccm_CreaseCount(cage),
                                  cage->creases,
                                  0);
}

bool LoadCageVertexPointBuffer(const cc_Mesh *cage)
{
    return LoadCatmullClarkBuffer(BUFFER_CAGE_VERTEX_POINTS,
                                  sizeof(cc_VertexPoint) * ccm_VertexCount(cage),
                                  cage->vertexPoints,
                                  0);
}

bool LoadCageVertexUvBuffer(const cc_Mesh *cage)
{
    if (ccm_UvCount(cage) > 0) {
        return LoadCatmullClarkBuffer(BUFFER_CAGE_VERTEX_UVS,
                                      sizeof(cc_VertexUv) * ccm_UvCount(cage),
                                      cage->uvs,
                                      0);
    } else {
        return true;
    }
}

bool LoadSubdVertexPointBuffer(const cc_Subd *subd)
{
    return LoadCatmullClarkBuffer(BUFFER_SUBD_VERTEX_POINTS,
                                  sizeof(cc_VertexPoint) * ccs_CumulativeVertexCount(subd),
                                  NULL,
                                  GL_MAP_READ_BIT);
}

bool LoadSubdHalfedgeBuffer(const cc_Subd *subd)
{
    return LoadCatmullClarkBuffer(BUFFER_SUBD_HALFEDGES,
                                  sizeof(cc_Halfedge_SemiRegular) * ccs_CumulativeHalfedgeCount(subd),
                                  NULL,
                                  GL_MAP_READ_BIT);
}

bool LoadSubdCreaseBuffer(const cc_Subd *subd)
{
    return LoadCatmullClarkBuffer(BUFFER_SUBD_CREASES,
                                  sizeof(cc_Crease) * ccs_CumulativeCreaseCount(subd),
                                  NULL,
                                  GL_MAP_READ_BIT);
}

bool LoadCageCounterBuffer(const cc_Mesh *cage)
{
    const struct {
        int32_t vertexCount;
        int32_t halfedgeCount;
        int32_t edgeCount;
        int32_t faceCount;
        int32_t uvCount;
    } counters = {
        ccm_VertexCount(cage),
        ccm_HalfedgeCount(cage),
        ccm_EdgeCount(cage),
        ccm_FaceCount(cage),
        ccm_UvCount(cage)
    };

    return LoadCatmullClarkBuffer(BUFFER_CAGE_COUNTERS,
                                  sizeof(counters),
                                  &counters,
                                  0);
}

bool LoadSubdMaxDepthBuffer(const cc_Subd *subd)
{
    const int32_t maxDepth = ccs_MaxDepth(subd);

    return LoadCatmullClarkBuffer(BUFFER_SUBD_MAXDEPTH,
                                  sizeof(maxDepth),
                                  &maxDepth,
                                  0);
}

bool LoadBuffers(const cc_Subd *subd)
{
    bool success = true;

    if (success) success = LoadCageVertexToHalfedgeIDsBuffer(subd->cage);
    if (success) success = LoadCageEdgeToHalfedgeIDsBuffer(subd->cage);
    if (success) success = LoadCageFaceToHalfedgeIDsBuffer(subd->cage);
    if (success) success = LoadCageHalfedgeBuffer(subd->cage);
    if (success) success = LoadCageCreaseBuffer(subd->cage);
    if (success) success = LoadCageVertexPointBuffer(subd->cage);
    if (success) success = LoadCageVertexUvBuffer(subd->cage);
    if (success) success = LoadCageCounterBuffer(subd->cage);
    if (success) success = LoadSubdVertexPointBuffer(subd);
    if (success) success = LoadSubdHalfedgeBuffer(subd);
    if (success) success = LoadSubdCreaseBuffer(subd);
    if (success) success = LoadSubdMaxDepthBuffer(subd);

    return success;
}


bool Load(const cc_Subd *subd)
{
    bool success = true;

    if (success) success = LoadPrograms();
    if (success) success = LoadBuffers(subd);

    return success;
}

void Release()
{
    glDeleteBuffers(BUFFER_COUNT, g_gl.buffers);

    for (int i = 0; i < PROGRAM_COUNT; ++i)
        glDeleteProgram(g_gl.programs[i]);
}


void CageSubdivisionCommand(int32_t programID, int32_t threadCount)
{
    const int32_t count = (threadCount >> g_subd.computeShaderLocalSize) + 1;

    glUseProgram(g_gl.programs[programID]);
    glDispatchCompute(count, 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    glUseProgram(0);
}

void SubdivisionCommand(int32_t programID, int32_t threadCount, int32_t depth)
{
    const int32_t count = (threadCount >> g_subd.computeShaderLocalSize) + 1;
    const int32_t uniformLocation =
            glGetUniformLocation(g_gl.programs[programID], "u_Depth");

    glUseProgram(g_gl.programs[programID]);
    glUniform1i(uniformLocation, depth);
    glDispatchCompute(count, 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    glUseProgram(0);
}

void RefineCageHalfedgesCommand(const cc_Subd *subd)
{
    CageSubdivisionCommand(PROGRAM_SUBD_CAGE_HALFEDGES,
                           ccm_HalfedgeCount(subd->cage));
}

void RefineCageCreasesCommand(const cc_Subd *subd)
{
    CageSubdivisionCommand(PROGRAM_SUBD_CAGE_CREASES,
                           ccm_CreaseCount(subd->cage));
}

#ifndef CC_DISABLE_UV
void RefineCageVertexUvsCommand(const cc_Subd *subd)
{
    CageSubdivisionCommand(PROGRAM_SUBD_CAGE_VERTEX_UVS,
                           ccm_HalfedgeCount(subd->cage));
}
#endif

void RefineCageFacesCommand(const cc_Subd *subd)
{
    CageSubdivisionCommand(PROGRAM_SUBD_CAGE_FACE_POINTS,
                           ccm_HalfedgeCount(subd->cage));
}

void RefineCageEdgesCommand(const cc_Subd *subd)
{
    CageSubdivisionCommand(PROGRAM_SUBD_CAGE_EDGE_POINTS,
                           ccm_HalfedgeCount(subd->cage));
}

void RefineCageVerticesCommand(const cc_Subd *subd)
{
    CageSubdivisionCommand(PROGRAM_SUBD_CAGE_VERTEX_POINTS,
                           ccm_HalfedgeCount(subd->cage));
}

void RefineHalfedgesCommand(const cc_Subd *subd, int32_t depth)
{
    SubdivisionCommand(PROGRAM_SUBD_HALFEDGES,
                       ccm_HalfedgeCountAtDepth(subd->cage, depth),
                       depth);
}

void RefineCreasesCommand(const cc_Subd *subd, int32_t depth)
{
    SubdivisionCommand(PROGRAM_SUBD_CREASES,
                       ccm_CreaseCountAtDepth(subd->cage, depth),
                       depth);
}

#ifndef CC_DISABLE_UV
void RefineVertexUvsCommand(const cc_Subd *subd, int32_t depth)
{
    SubdivisionCommand(PROGRAM_SUBD_VERTEX_UVS,
                       ccm_HalfedgeCountAtDepth(subd->cage, depth),
                       depth);
}
#endif

void RefineFacesCommand(const cc_Subd *subd, int32_t depth)
{
    SubdivisionCommand(PROGRAM_SUBD_FACE_POINTS,
                       ccm_HalfedgeCountAtDepth(subd->cage, depth),
                       depth);
}

void RefineEdgesCommand(const cc_Subd *subd, int32_t depth)
{
    SubdivisionCommand(PROGRAM_SUBD_EDGE_POINTS,
                       ccm_HalfedgeCountAtDepth(subd->cage, depth),
                       depth);
}

void RefineVertexPointsCommand(const cc_Subd *subd, int32_t depth)
{
    SubdivisionCommand(PROGRAM_SUBD_VERTEX_POINTS,
                       ccm_HalfedgeCountAtDepth(subd->cage, depth),
                       depth);
}

void RefineVertexPoints(const cc_Subd *subd)
{
    glClearNamedBufferData(g_gl.buffers[BUFFER_SUBD_VERTEX_POINTS],
                           GL_R32F,
                           GL_RED,
                           GL_FLOAT,
                           NULL);
    RefineCageFacesCommand(subd);
    RefineCageEdgesCommand(subd);
    RefineCageVerticesCommand(subd);

    for (int32_t depth = 1; depth < ccs_MaxDepth(subd); ++depth) {
        RefineFacesCommand(subd, depth);
        RefineEdgesCommand(subd, depth);
        RefineVertexPointsCommand(subd, depth);
    }
}

void RefineHalfedges(const cc_Subd *subd)
{
    RefineCageHalfedgesCommand(subd);

    for (int32_t depth = 1; depth < ccs_MaxDepth(subd); ++depth) {
        RefineHalfedgesCommand(subd, depth);
    }
}

void RefineCreases(const cc_Subd *subd)
{
    RefineCageCreasesCommand(subd);

    for (int32_t depth = 1; depth < ccs_MaxDepth(subd); ++depth) {
        RefineCreasesCommand(subd, depth);
    }
}

#ifndef CC_DISABLE_UV
void RefineVertexUvs(const cc_Subd *subd)
{
    RefineCageVertexUvsCommand(subd);

    for (int32_t depth = 1; depth < ccs_MaxDepth(subd); ++depth) {
        RefineVertexUvsCommand(subd, depth);
    }
}
#endif

typedef struct {
    double min, max, median, mean;
} BenchStats;

static int CompareCallback(const void * a, const void * b)
{
    if (*(double*)a > *(double*)b) {
        return 1;
    } else if (*(double*)a < *(double*)b) {
        return -1;
    } else {
        return 0;
    }
}

BenchStats Bench(void (*SubdCallback)(const cc_Subd *subd), const cc_Subd *subd)
{
#ifdef FLAG_BENCH
    const int32_t runCount = 100;
#else
    const int32_t runCount = 1;
#endif
    double *times = (double *)malloc(sizeof(*times) * runCount);
    double timesTotal = 0.0;
    BenchStats stats = {0.0, 0.0, 0.0, 0.0};
    djg_clock *clock = djgc_create();

    for (int32_t runID = 0; runID < runCount; ++runID) {
        double cpuTime = 0.0, gpuTime = 0.0;

        glFinish();
        djgc_start(clock);
        (*SubdCallback)(subd);
        djgc_stop(clock);
        glFinish();
        djgc_ticks(clock, &cpuTime, &gpuTime);

        times[runID] = gpuTime;
        timesTotal+= gpuTime;
    }

    qsort(times, runCount, sizeof(times[0]), &CompareCallback);

    stats.min = times[0];
    stats.max = times[runCount - 1];
    stats.median = times[runCount / 2];
    stats.mean = timesTotal / runCount;

    free(times);
    djgc_release(clock);

    return stats;
}

void GetHalfedges(cc_Subd *subd)
{
    const GLuint *buffer = &g_gl.buffers[BUFFER_SUBD_HALFEDGES];
    const cc_Halfedge_SemiRegular *halfedges =
            (cc_Halfedge_SemiRegular *)glMapNamedBuffer(*buffer, GL_READ_ONLY);

    memcpy(subd->halfedges,
           halfedges,
           ccs_CumulativeHalfedgeCount(subd) * sizeof(cc_Halfedge_SemiRegular));

    glUnmapNamedBuffer(*buffer);
}

void GetVertices(cc_Subd *subd)
{
    const GLuint *buffer = &g_gl.buffers[BUFFER_SUBD_VERTEX_POINTS];
    const cc_VertexPoint *vertices =
            (cc_VertexPoint *)glMapNamedBuffer(*buffer, GL_READ_ONLY);

    memcpy(subd->vertexPoints,
           vertices,
           ccs_CumulativeVertexCount(subd) * sizeof(cc_VertexPoint));

    glUnmapNamedBuffer(*buffer);
}

void GetCreases(cc_Subd *subd)
{
    const GLuint *buffer = &g_gl.buffers[BUFFER_SUBD_CREASES];
    const cc_Crease *creases =
            (cc_Crease *)glMapNamedBuffer(*buffer, GL_READ_ONLY);

    memcpy(subd->creases,
           creases,
           ccs_CumulativeCreaseCount(subd) * sizeof(cc_Crease));

    glUnmapNamedBuffer(*buffer);
}


/*******************************************************************************
 * ExportToObj -- Exports subd to the OBJ file format
 *
 */
static void
ExportToObj(
    const cc_Subd *subd,
    int32_t depth,
    const char *filename
) {
    const cc_Mesh *cage = subd->cage;
    const int32_t vertexPointCount = ccm_VertexCountAtDepth(cage, depth);
    const int32_t faceCount = ccm_FaceCountAtDepth(cage, depth);
    FILE *pf = fopen(filename, "w");

    // write vertices
    fprintf(pf, "# Vertices\n");
    if (depth == 0) {
        const int32_t vertexUvCount = ccm_UvCount(cage);

        for (int32_t vertexID = 0; vertexID < vertexPointCount; ++vertexID) {
            const float *v = ccm_VertexPoint(cage, vertexID).array;

            fprintf(pf, "v %f %f %f\n", v[0], v[1], v[2]);
        }

        for (int32_t vertexID = 0; vertexID < vertexUvCount; ++vertexID) {
            const float *v = ccm_Uv(cage, vertexID).array;

            fprintf(pf, "vt %f %f\n", v[0], v[1]);
        }
    } else {
        const int32_t halfedgeCount = ccm_HalfedgeCountAtDepth(cage, depth);

        for (int32_t vertexID = 0; vertexID < vertexPointCount; ++vertexID) {
            const float *v = ccs_VertexPoint(subd, vertexID, depth).array;

            fprintf(pf, "v %f %f %f\n", v[0], v[1], v[2]);
        }

#ifndef CC_DISABLE_UV
        for (int32_t halfedgeID = 0; halfedgeID < halfedgeCount; ++halfedgeID) {
            const float *uv = ccs_HalfedgeVertexUv(subd, halfedgeID, depth).array;

            fprintf(pf, "vt %f %f\n", uv[0], uv[1]);
        }
#endif
    }
    fprintf(pf, "\n");

    // write topology
    fprintf(pf, "# Topology\n");
    if (depth == 0) {
        for (int32_t faceID = 0; faceID < faceCount; ++faceID) {
            const int32_t halfEdgeID = ccm_FaceToHalfedgeID(cage, faceID);

            fprintf(pf,
                    "f %i/%i",
                    ccm_HalfedgeVertexID(cage, halfEdgeID) + 1,
                    ccm_HalfedgeUvID(cage, halfEdgeID) + 1);

            for (int32_t halfEdgeIt = ccm_HalfedgeNextID(cage, halfEdgeID);
                         halfEdgeIt != halfEdgeID;
                         halfEdgeIt = ccm_HalfedgeNextID(cage, halfEdgeIt)) {
                fprintf(pf,
                        " %i/%i",
                        ccm_HalfedgeVertexID(cage, halfEdgeIt) + 1,
                        ccm_HalfedgeUvID(cage, halfEdgeIt) + 1);
            }
            fprintf(pf, "\n");
        }
    } else {
        for (int32_t faceID = 0; faceID < faceCount; ++faceID) {
#ifndef CC_DISABLE_UV
            fprintf(pf,
                    "f %i/%i %i/%i %i/%i %i/%i\n",
                    ccs_HalfedgeVertexID(subd, 4 * faceID + 0, depth) + 1,
                    4 * faceID + 1,
                    ccs_HalfedgeVertexID(subd, 4 * faceID + 1, depth) + 1,
                    4 * faceID + 2,
                    ccs_HalfedgeVertexID(subd, 4 * faceID + 2, depth) + 1,
                    4 * faceID + 3,
                    ccs_HalfedgeVertexID(subd, 4 * faceID + 3, depth) + 1,
                    4 * faceID + 4);
#else
            fprintf(pf,
                    "f %i %i %i %i\n",
                    ccs_HalfedgeVertexID(subd, 4 * faceID + 0, depth) + 1,
                    ccs_HalfedgeVertexID(subd, 4 * faceID + 1, depth) + 1,
                    ccs_HalfedgeVertexID(subd, 4 * faceID + 2, depth) + 1,
                    ccs_HalfedgeVertexID(subd, 4 * faceID + 3, depth) + 1);
#endif
        }
        fprintf(pf, "\n");
    }

    fclose(pf);
}


int main(int argc, char **argv)
{
    const char *filename = "./ArmorGuy.ccm";
    int32_t maxDepth = 4;
#ifdef FLAG_BENCH
    int32_t exportToObj = 0;
#else
    int32_t exportToObj = 1;
#endif
    cc_Mesh *cage = NULL;
    cc_Subd *subd = NULL;

    if (argc > 1) {
        filename = argv[1];
    }

    if (argc > 2) {
        maxDepth = atoi(argv[2]);
    }

    if (argc > 3) {
        exportToObj = atoi(argv[3]);
    }

    cage = ccm_Load(filename);

    if (!cage) {
        return -1;
    }

    subd = ccs_Create(cage, maxDepth);

    if (!subd) {
        ccm_Release(cage);

        return -1;
    }

    LOG("Loading {OpenGL Window}");
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, g_window.glversion.major);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, g_window.glversion.minor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
#ifndef NDEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif
    g_window.handle = glfwCreateWindow(g_window.width,
                                       g_window.height,
                                       g_window.name,
                                       NULL, NULL);
    if (g_window.handle == NULL) {
        LOG("=> Failure <=");
        glfwTerminate();

        return -1;
    }
    glfwMakeContextCurrent(g_window.handle);

    // load OpenGL functions
    LOG("Loading {OpenGL Functions}");
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        LOG("=> Failure <=");
        glfwTerminate();

        return -1;
    }

    // initialize
    LOG("Loading {Demo}");
    if (!Load(subd)) {
        LOG("=> Failure <=");
        glfwTerminate();

        return -1;
    }

    // execute GPU kernels
    {
        const BenchStats stats = Bench(&RefineCreases, subd);

        LOG("Creases      -- median/mean/min/max (ms): %f / %f / %f / %f",
            stats.median * 1e3,
            stats.mean * 1e3,
            stats.min * 1e3,
            stats.max * 1e3);
    }

    {
        const BenchStats stats = Bench(&RefineHalfedges, subd);

        LOG("Halfedges    -- median/mean/min/max (ms): %f / %f / %f / %f",
            stats.median * 1e3,
            stats.mean * 1e3,
            stats.min * 1e3,
            stats.max * 1e3);
    }

    {
        const BenchStats stats = Bench(&RefineVertexPoints, subd);

        LOG("VertexPoints -- median/mean/min/max (ms): %f / %f / %f / %f",
            stats.median * 1e3,
            stats.mean * 1e3,
            stats.min * 1e3,
            stats.max * 1e3);
    }

#ifndef CC_DISABLE_UV
    {
        const BenchStats stats = Bench(&RefineVertexUvs, subd);

        LOG("VertexUvs    -- median/mean/min/max (ms): %f / %f / %f / %f",
            stats.median * 1e3,
            stats.mean * 1e3,
            stats.min * 1e3,
            stats.max * 1e3);
    }
#endif

    if (exportToObj > 0) {
        LOG("Exporting...");
        GetHalfedges(subd);
        GetVertices(subd);
        GetCreases(subd);

        for (int32_t depth = 0; depth <= ccs_MaxDepth(subd); ++depth) {
            char buf[64];

            sprintf(buf, "subd_%01i_gpu.obj", depth);
            ExportToObj(subd, depth, buf);
            LOG("Level %i: done.", depth);
        }
    }

    LOG("All done!");

    Release();
    glfwTerminate();
    ccm_Release(cage);
    ccs_Release(subd);

    return 0;
}
