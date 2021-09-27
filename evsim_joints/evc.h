#ifndef GPU_H
#define GPU_H

#include <cstdint>
#include <cstdio>

#define GLEW_STATIC
#include <GL/glew.h>
#include <SDL2/SDL.h>

struct evcData_t {
    int width, height;

    // cuda output intermediate array
    GLubyte *cudaDestPtr;

    // cuda output texture
    GLuint cudaTexResult;
    // cuda output texture resource
    void *cudaTexResultResource;

    // cuda backbuffer texture
    GLuint cudaTexInput;
    // cuda backbuffer resource
    void *cudaTexInputResource;

    // internal stuff begins here
    // cuda camera memory array
    float *cudaMemoryFramePtr;
    // cuda compact output array
    GLubyte *cudaCompactPtr;
    // host compact output array
    GLubyte *hostCompactPtr;

    // cuda uniform random event noise array
    float *cudaNoisePtr;
    // curand prng
    void  *curandGenerator;

    // file writer data
    FILE *file;
    struct fileBuf_t{
        // because 320>255
        unsigned short x;
        // because 240<255
        unsigned char y;
        // 1 - positive, 2 - negative, 255 - new frame
        unsigned char p;
    };
    fileBuf_t *buf;
    size_t bufCap;
};

void init_evc(evcData_t *evc, GLuint cudaTexInput, int width, int height, const char *fileName);

void destroy_evc(evcData_t *evc);

//void register_image(void **cudaTexInputResource, GLuint texture);
//void unregister_image(void *cudaTexInputResource);

void cuda_draw(evcData_t *evc, bool screenRender, float c);

#endif // GPU_H
