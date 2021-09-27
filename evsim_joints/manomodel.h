#ifndef MANOMODEL_H
#define MANOMODEL_H

#include <glm/glm.hpp>

#include "model.h"
#include "shader.h"

struct ManoModel : public Model {
    std::vector<float> normals, weights, texcoords;
    std::vector<glm::mat4x3> A;
    GLuint nbo, wbo, tbo;

    ManoModel(const std::vector<float> &_vertices,
              const std::vector<float> &_normals,
              const std::vector<GLuint> &_elements,
              const std::vector<float> &_weights,
              const std::vector<float> &_texcoords,
              const std::vector<glm::mat4x3> &_A,
              const std::function<void (GLuint, GLuint, GLuint, GLuint)> &setup_attribs);

    virtual ~ManoModel();

    void update_shader(Shader &shader);
    void update_normals();
};

#endif // MANOMODEL_H
