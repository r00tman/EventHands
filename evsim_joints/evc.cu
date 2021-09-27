#include "evc.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <curand.h>

__global__
void event_camera_kernel(GLubyte *buf, GLubyte *compact, float *memory,
                                    cudaTextureObject_t inTex,
                                    float *noise,
                                    int width, int height, float c) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    uint idx = width * y + x;
    uint pos = 4*idx;

    uchar4 pxl = tex2D<uchar4>(inTex, x, y);

    const float EPS = 1;

    float next = log(pxl.x*0.2+pxl.y*0.7+pxl.z*0.1+EPS);
//    float next = pxl.x*0.2+pxl.y*0.7+pxl.z*0.1;
    float prev = memory[idx];

//    float diff = log((next+EPS)/(prev+EPS));
    float diff = next - prev;

    compact[idx] = 0;

    if (buf) {
        buf[pos+0] = buf[pos+1] = buf[pos+2] = 0; // rgb
        buf[pos+3] = 255; // alpha
    }

    if (diff > c) {
        if (buf) {
            buf[pos+0] = 255;
        }
        int count = max(1.0f, floor(diff/c));
        compact[idx] = 2*count+1-2;
        memory[idx] = prev+count*c;
//        memory[idx] = floor(next/c)*c;
    }
    if (diff < -c) {
        if (buf) {
            buf[pos+2] = 255;
        }
        int count = max(1.0f, floor(-diff/c));
        compact[idx] = 2*count+2-2;
        memory[idx] = prev-count*c;
//        memory[idx] = ceil(next/c)*c;
    }
//    const float THR = 0.000001;
    const float POSTHR = 5.5e-5;
    const float NEGTHR = 5.5e-5*100/2500;
    if (noise[idx] > 1-POSTHR) {
        if (buf) {
            buf[pos+0] = 255;
            buf[pos+2] = 0;
        }
        compact[idx] = 1;
    }
    if (noise[idx] < NEGTHR) {
        if (buf) {
            buf[pos+0] = 0;
            buf[pos+2] = 255;
        }
        compact[idx] = 2;
    }
//    buf[pos+1] = cnoise[idx]*127;
}

void render(evcData_t *evc, cudaTextureObject_t inTex, bool screenRender, float c) {
    const dim3 threadsPerBlock(16, 16);
    const dim3 blocksPerGrid((evc->width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                             (evc->height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    curandGenerateUniform(*(curandGenerator_t*)evc->curandGenerator, evc->cudaNoisePtr, evc->width*evc->height);

    GLubyte *cudaDestPtr = screenRender ? evc->cudaDestPtr : NULL;
    event_camera_kernel<<<blocksPerGrid, threadsPerBlock>>>(cudaDestPtr, evc->cudaCompactPtr,
                                                            evc->cudaMemoryFramePtr, inTex,
                                                            evc->cudaNoisePtr,
                                                            evc->width, evc->height, c);
}

void register_image(void **cudaTexInputResource, GLuint texture) {
     cudaGraphicsGLRegisterImage((cudaGraphicsResource**)cudaTexInputResource, texture,
                                 GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
}

void unregister_image(void *cudaTexInputResource) {
//    cudaGraphicsUnregisterResource((cudaGraphicsResource_t*)cudaTexInputResource);
}

void init_evc(evcData_t *evc, GLuint cudaTexInput, int width, int height, const char *fileName) {
    evc->width = width;
    evc->height = height;
    evc->cudaTexInput = cudaTexInput;

    cudaMalloc(&evc->cudaDestPtr, evc->width*evc->height*4);

    glGenTextures(1, &evc->cudaTexResult);
    glBindTexture(GL_TEXTURE_2D, evc->cudaTexResult);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, evc->width, evc->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    register_image(&evc->cudaTexResultResource, evc->cudaTexResult);
    register_image(&evc->cudaTexInputResource, evc->cudaTexInput);

    // init internal stuff
    cudaMalloc(&evc->cudaCompactPtr, evc->width*evc->height);
    cudaMallocHost(&evc->hostCompactPtr, evc->width*evc->height);
    {
        cudaMalloc(&evc->cudaMemoryFramePtr, sizeof(float)*evc->width*evc->height);
        thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(evc->cudaMemoryFramePtr);
        thrust::fill(dev_ptr, dev_ptr + evc->width*evc->height, 0.0f);
    }
    cudaMalloc(&evc->cudaNoisePtr, evc->width*evc->height*sizeof(float));

    evc->curandGenerator = (void*)new curandGenerator_t;
    curandCreateGenerator((curandGenerator_t*)evc->curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(*(curandGenerator_t*)evc->curandGenerator, 1234);

    if (fileName) {
        evc->file = fopen(fileName, "wb");
    } else {
        evc->file = NULL;
    }
    evc->bufCap = evc->width*evc->height*64+1;
    evc->buf = new evcData_t::fileBuf_t[evc->bufCap];
}

void destroy_evc(evcData_t *evc) {
    unregister_image(evc->cudaTexInputResource);
    unregister_image(evc->cudaTexResultResource);
    glDeleteTextures(1, &evc->cudaTexResult);
    cudaFree(evc->cudaDestPtr);

    // free internal stuff
    cudaFree(evc->cudaMemoryFramePtr);
    cudaFree(evc->cudaCompactPtr);
    cudaFreeHost(evc->hostCompactPtr);

    cudaFree(evc->cudaNoisePtr);

    curandDestroyGenerator(*(curandGenerator_t*)evc->curandGenerator);
    delete (curandGenerator_t*)evc->curandGenerator;

    delete[] evc->buf;
    if (evc->file) {
        fclose(evc->file);
    }
}

void write_frame(evcData_t *evc) {
    if (!evc->file) {
        return;
    }
    int height = evc->height, width = evc->width;
    GLubyte *compactPtr = evc->hostCompactPtr;
    evcData_t::fileBuf_t *buf = evc->buf;
    int bufLen = 0;

    buf[bufLen].x = 0;     // new frame tag
    buf[bufLen].y = 0;     // new frame tag
    buf[bufLen++].p = 255; // new frame tag
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            short b = compactPtr[i*width+j];
            unsigned char p = b%2;
            while (b > 0) {
                buf[bufLen].x = j;
                buf[bufLen].y = height-i-1;
                buf[bufLen++].p = p;
                b -= 2;
            }
        }
    }
    assert(bufLen <= evc->bufCap);
    fwrite_unlocked(buf, sizeof(buf[0]), bufLen, evc->file);
}

void cuda_draw(evcData_t *evc, bool screenRender, float c) {
    cudaArray *inputArray;

    cudaGraphicsResource *resources[2] = {
        (cudaGraphicsResource*)evc->cudaTexInputResource,
        (cudaGraphicsResource*)evc->cudaTexResultResource
    };
    cudaGraphicsMapResources(screenRender ? 2 : 1, resources, 0);
    cudaGraphicsSubResourceGetMappedArray(&inputArray, (cudaGraphicsResource*)evc->cudaTexInputResource, 0, 0);

    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, inputArray);

    cudaTextureObject_t inTexObject;
    {
        cudaResourceDesc texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));

        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array = inputArray;

        cudaTextureDesc texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));

        texDescr.normalizedCoords = false;
        texDescr.filterMode = cudaFilterModePoint;
        texDescr.addressMode[0] = cudaAddressModeWrap;
        texDescr.readMode = cudaReadModeElementType;

        cudaCreateTextureObject(&inTexObject, &texRes, &texDescr, NULL);
    }

    render(evc, inTexObject, screenRender, c);

    cudaMemcpy(evc->hostCompactPtr, evc->cudaCompactPtr, evc->width*evc->height, cudaMemcpyDeviceToHost);
    write_frame(evc);

    cudaDestroyTextureObject(inTexObject);

    if (screenRender) {
        cudaArray *texturePtr;
        cudaGraphicsSubResourceGetMappedArray(&texturePtr, (cudaGraphicsResource*)evc->cudaTexResultResource, 0, 0);
        cudaMemcpy2DToArray(texturePtr, 0, 0, evc->cudaDestPtr, evc->width*4, evc->width*4, evc->height, cudaMemcpyDeviceToDevice);
    }

    cudaGraphicsUnmapResources(screenRender ? 2 : 1, resources, 0);
}
