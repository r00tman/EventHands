#include "shader.h"

#include <iostream>
#include <fstream>

std::string read_file(const std::string &fn) {
    std::ifstream file(fn);
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0);
    file.read(&buffer[0], size);
    return buffer;
}

GLuint load_shader(const std::string &fn, GLenum shader_type) {
    std::string source_str = read_file(fn);

    GLuint shader = glCreateShader(shader_type);
    const char *source = source_str.c_str();
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    char buffer[16384];
    glGetShaderInfoLog(shader, sizeof(buffer), NULL, buffer);
    std::cerr << status << " " << buffer << std::endl;
    return shader;
}

Shader::Shader(const std::string &vert_fn, const std::string &frag_fn) {
    vertShader = load_shader(vert_fn, GL_VERTEX_SHADER);
    fragShader = load_shader(frag_fn, GL_FRAGMENT_SHADER);
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertShader);
    glAttachShader(shaderProgram, fragShader);

    glBindFragDataLocation(shaderProgram, 0, "outColor");

    glLinkProgram(shaderProgram);
}

Shader::~Shader() {
    glDeleteProgram(shaderProgram);
    glDeleteShader(fragShader);
    glDeleteShader(vertShader);
}

void Shader::use() {
    glUseProgram(shaderProgram);
}

GLuint Shader::attrib(const std::string &attrib) {
    return glGetAttribLocation(shaderProgram, attrib.c_str());
}

GLuint Shader::operator[](const std::string &uniform) {
    auto el = uniforms.find(uniform);
    if (el != uniforms.end()) {
        return el->second;
    }
    GLuint res = glGetUniformLocation(shaderProgram, uniform.c_str());
    uniforms.insert(make_pair(uniform, res));
    return res;
}
