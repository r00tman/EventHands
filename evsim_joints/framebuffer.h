#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#define GLEW_STATIC
#include <GL/glew.h>

struct Framebuffer {
    GLuint frameBuffer;
    GLuint texColorBuffer;
    GLuint rboDepth;

    Framebuffer(int width, int height);

    ~Framebuffer();

    Framebuffer(const Framebuffer&) = delete;
    void operator=(const Framebuffer&) = delete;

    void use();

    static void use_screen_fb();
};

#endif // FRAMEBUFFER_H
