#ifndef TEXTURE_H
#define TEXTURE_H

#include <string>

#define GLEW_STATIC
#include <GL/glew.h>

#include "shader.h"

struct Texture {
    GLuint tex;

    Texture();
    Texture(const std::string &fn);

    void use(GLuint uniform, GLuint unit = 0);

    virtual ~Texture();

    Texture(const Texture &) = delete;
    void operator=(const Texture &) = delete;
};

#endif // TEXTURE_H
