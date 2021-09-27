#ifndef MANO_GPU_H
#define MANO_GPU_H

#define GLEW_STATIC
#include <GL/glew.h>

void compute_normals_gpu(float *gpuVertices, float *gpuNormals, GLuint *gpuElements, int vertexCount, int triangleCount);
//void compute_normals(float *vertices, GLuint *elements, int vertexCount, int triangleCount);

struct manoData_t {
    // vertex buffer
    int vertexCount;
    GLuint vbo;
    void *cudaVboResource;

    // normal buffer
    GLuint nbo;
    void *cudaNboResource;

    // index buffer
    int triangleCount;
    GLuint ebo;
    void *cudaEboResource;

    float *cudaVShaped;

    float *cudaPoseDirs;

    float *hostPosemapped;
    float *cudaPosemapped;

    void *cublasHandle;
};

void init_mano_gpu(manoData_t *data, GLuint vbo, GLuint nbo, GLuint ebo, float *vShaped, float *poseDirs, int vertexCount, int triangleCount);
void destroy_mano_gpu(manoData_t *data);

void update_mano_vshaped(manoData_t *data, float *vShaped);
void compute_mano_and_normals_vbo(manoData_t *data);

#endif // MANO_GPU_H
