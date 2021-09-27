#include "manogpu.h"

#include <cstdio>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cublas_v2.h>

__global__
void compute_normals_kernel(float *vertices, float *normals, GLuint *elements, int triangleCount) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= triangleCount) {
        return;
    }
    int ai = 3*elements[3*i+0];
    int bi = 3*elements[3*i+1];
    int ci = 3*elements[3*i+2];

    float *a = &vertices[ai];
    float *b = &vertices[bi];
    float *c = &vertices[ci];

    float x[3] = {b[0]-a[0], b[1]-a[1], b[2]-a[2]};
    float y[3] = {c[0]-a[0], c[1]-a[1], c[2]-a[2]};

    float p[3];
    p[0] = x[1]*y[2]-y[1]*x[2];
    p[1] = x[2]*y[0]-y[2]*x[0];
    p[2] = x[0]*y[1]-y[0]*x[1];

    atomicAdd(normals+ai+0, p[0]);
    atomicAdd(normals+ai+1, p[1]);
    atomicAdd(normals+ai+2, p[2]);

    atomicAdd(normals+bi+0, p[0]);
    atomicAdd(normals+bi+1, p[1]);
    atomicAdd(normals+bi+2, p[2]);

    atomicAdd(normals+ci+0, p[0]);
    atomicAdd(normals+ci+1, p[1]);
    atomicAdd(normals+ci+2, p[2]);
}

__global__
void normalize_normals_kernel(float *normals, int vertexCount) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= vertexCount) {
        return;
    }
    const int offset = 3*i;
    const float x = normals[offset+0];
    const float y = normals[offset+1];
    const float z = normals[offset+2];
    const float len = sqrtf(x*x+y*y+z*z);
//    const float ilen = (len>0.001)?1.0f/len:0.0f;
    const float ilen = 1/len;
    normals[offset+0] *= ilen;
    normals[offset+1] *= ilen;
    normals[offset+2] *= ilen;
}

__global__
void zero_normals_kernel(float *normals, int vertexCount) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= vertexCount) {
        return;
    }
    const int offset = 3*i;
    normals[offset+0] = 0.0f;
    normals[offset+1] = 0.0f;
    normals[offset+2] = 0.0f;
}

void compute_normals_gpu(float *gpuVertices, float *gpuNormals, GLuint *gpuElements, int vertexCount, int triangleCount) {
    const dim3 threadsPerBlock(32, 1);
    {
        const dim3 blocksPerGrid((vertexCount + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);
        zero_normals_kernel<<<blocksPerGrid, threadsPerBlock>>>(gpuNormals, vertexCount);
    }
    {
        const dim3 blocksPerGrid((triangleCount + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);
        compute_normals_kernel<<<blocksPerGrid, threadsPerBlock>>>(gpuVertices, gpuNormals, gpuElements, triangleCount);
    }
    {
        const dim3 blocksPerGrid((vertexCount + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);
        normalize_normals_kernel<<<blocksPerGrid, threadsPerBlock>>>(gpuNormals, vertexCount);
    }
}

//void compute_normals(float *vertices, GLuint *elements, int vertexCount, int triangleCount) {
//    float *gpuVertices;
//    GLuint *gpuElements;
//    cudaMalloc(&gpuVertices, 6*sizeof(float)*vertexCount);
//    cudaMalloc(&gpuElements, 3*sizeof(GLuint)*triangleCount);

//    cudaMemcpy(gpuVertices, vertices, 6*sizeof(float)*vertexCount, cudaMemcpyHostToDevice);
//    cudaMemcpy(gpuElements, elements, 3*sizeof(GLuint)*triangleCount, cudaMemcpyHostToDevice);

//    compute_normals_gpu(gpuVertices, gpuElements, vertexCount, triangleCount);

//    cudaMemcpy(vertices, gpuVertices, 6*sizeof(float)*vertexCount, cudaMemcpyDeviceToHost);

//    cudaFree(gpuVertices);
//    cudaFree(gpuElements);
//}

void init_mano_gpu(manoData_t *data, GLuint vbo, GLuint nbo, GLuint ebo, float *vShaped, float *poseDirs, int vertexCount, int triangleCount) {
    data->vertexCount = vertexCount;
    data->triangleCount = triangleCount;

    data->vbo = vbo;
    cudaGraphicsGLRegisterBuffer((cudaGraphicsResource**)&data->cudaVboResource, data->vbo, cudaGraphicsMapFlagsWriteDiscard);

    data->nbo = nbo;
    cudaGraphicsGLRegisterBuffer((cudaGraphicsResource**)&data->cudaNboResource, data->nbo, cudaGraphicsMapFlagsWriteDiscard);

    data->ebo = ebo;
    cudaGraphicsGLRegisterBuffer((cudaGraphicsResource**)&data->cudaEboResource, data->ebo, cudaGraphicsMapFlagsReadOnly);

    cudaMalloc(&data->cudaVShaped, data->vertexCount*3*sizeof(float));
    cudaMalloc(&data->cudaPoseDirs, data->vertexCount*3*459*sizeof(float));
    cudaMalloc(&data->cudaPosemapped, 459*sizeof(float));

    cudaMemcpy(data->cudaVShaped, vShaped, data->vertexCount*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data->cudaPoseDirs, poseDirs, data->vertexCount*3*459*sizeof(float), cudaMemcpyHostToDevice);

    data->cublasHandle = (void*)new cublasHandle_t;
    cublasCreate((cublasHandle_t*)data->cublasHandle);
}

void destroy_mano_gpu(manoData_t *data) {
    cublasDestroy(*(cublasHandle_t*)data->cublasHandle);
    delete (cublasHandle_t*)data->cublasHandle;

    cudaFree(data->cudaPoseDirs);
    cudaFree(data->cudaPosemapped);

    cudaGraphicsUnregisterResource((cudaGraphicsResource*)data->cudaVboResource);
    cudaGraphicsUnregisterResource((cudaGraphicsResource*)data->cudaNboResource);
    cudaGraphicsUnregisterResource((cudaGraphicsResource*)data->cudaEboResource);
}

void update_mano_vshaped(manoData_t *data, float *vShaped) {
    cudaMemcpy(data->cudaVShaped, vShaped, data->vertexCount*3*sizeof(float), cudaMemcpyHostToDevice);
}

__global__
void mult_add(float *A, float *x, float *output, int rows, int cols) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= rows) {
        return;
    }

    float res = output[i];
    for (int j = 0; j < cols; ++j) {
        res += A[i*cols+j]*x[j];
    }
    output[i] = res;
}

void compute_mano_and_normals_vbo(manoData_t *data) {
    cudaGraphicsResource *resources[] = {
        (cudaGraphicsResource*)data->cudaVboResource,
        (cudaGraphicsResource*)data->cudaNboResource,
        (cudaGraphicsResource*)data->cudaEboResource
    };
    cudaGraphicsMapResources(3, resources, 0);

    float *vertices, *normals;
    GLuint *elements;

    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&vertices, &num_bytes, resources[0]);
    cudaGraphicsResourceGetMappedPointer((void**)&normals, &num_bytes, resources[1]);
    cudaGraphicsResourceGetMappedPointer((void**)&elements, &num_bytes, resources[2]);

    cudaMemcpy(vertices, data->cudaVShaped, sizeof(float)*data->vertexCount*3, cudaMemcpyDeviceToDevice);
    cudaMemcpy(data->cudaPosemapped, data->hostPosemapped, sizeof(float)*459, cudaMemcpyHostToDevice);

//    const int tpb = 32;
//    const int bpg = (data->vertexCount*3+31)/tpb;
//    mult_add<<<bpg, tpb>>>(data->cudaPoseDirs, data->cudaPosemapped, vertices, data->vertexCount*3, 459);
    float one = 1.0f;
    float deform_weight = 1.0f;
    cublasSgemv(*(cublasHandle_t*)data->cublasHandle, CUBLAS_OP_T,
                459, data->vertexCount*3, &deform_weight, data->cudaPoseDirs, 459,
                data->cudaPosemapped, 1, &one, vertices, 1);

    compute_normals_gpu(vertices, normals, elements, data->vertexCount, data->triangleCount);

    cudaGraphicsUnmapResources(3, resources, 0);
}
