#ifndef SHADER_H
#define SHADER_H

#include <string>
#include <unordered_map>

#define GLEW_STATIC
#include <GL/glew.h>

struct Shader {
    GLuint vertShader, fragShader;
    GLuint shaderProgram;
    std::unordered_map<std::string, GLuint> uniforms;

    Shader(const std::string &vert_fn, const std::string &frag_fn);
    ~Shader();

    Shader(const Shader&) = delete;
    void operator=(const Shader&) = delete;

    void use();

    // find attribute
    GLuint attrib(const std::string &attrib);

    // find uniform
    GLuint operator[](const std::string &uniform);
};

#endif // SHADER_H
