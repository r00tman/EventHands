#include "manomodel.h"

#include <glm/gtc/type_ptr.hpp>

ManoModel::ManoModel(const std::vector<float> &_vertices,
                     const std::vector<float> &_normals,
                     const std::vector<GLuint> &_elements,
                     const std::vector<float> &_weights,
                     const std::vector<float> &_texcoords,
                     const std::vector<glm::mat4x3> &_A,
                     const std::function<void (GLuint, GLuint, GLuint, GLuint)> &setup_attribs) :
    Model(_vertices, _elements, true, [&] {})
{
    normals = _normals;
    weights = _weights;
    texcoords = _texcoords;
    A = _A;
    glBindVertexArray(vao);
    glGenBuffers(1, &nbo);
    glBindBuffer(GL_ARRAY_BUFFER, nbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*normals.size(), normals.data(), GL_DYNAMIC_DRAW);

    glGenBuffers(1, &wbo);
    glBindBuffer(GL_ARRAY_BUFFER, wbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*weights.size(), weights.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &tbo);
    glBindBuffer(GL_ARRAY_BUFFER, tbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*texcoords.size(), texcoords.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    setup_attribs(vbo, nbo, wbo, tbo);
    glBindVertexArray(0);
}

ManoModel::~ManoModel() {
    glDeleteBuffers(1, &nbo);
    glDeleteBuffers(1, &wbo);
    // and base destructor then deletes ebo, vbo, vao
}

void ManoModel::update_shader(Shader &shader) {
    glUniformMatrix4x3fv(shader["A"], A.size(), GL_FALSE, glm::value_ptr(A[0]));
}

void ManoModel::update_normals() {
    glBindBuffer(GL_ARRAY_BUFFER, nbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*normals.size(), normals.data());
}
