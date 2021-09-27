#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <string>
#include <memory>
#include <functional>

#define GLEW_STATIC
#include <GL/glew.h>

struct Model {
    std::vector<float> vertices;
    std::vector<GLuint> elements;

    GLuint vao;
    GLuint vbo;
    GLuint ebo;

    Model(const std::vector<float> &_vertices, const std::vector<GLuint> &_elements, bool dynamic,
          const std::function<void()> &setup_attribs);
    virtual ~Model();

    Model(const Model&) = delete;
    void operator=(const Model&) = delete;

    virtual void use();

    // draws all elements. use() should be called before
    virtual void draw();

    void update_vertices();
};

void load_obj(const std::string &fn, std::vector<float> &vertices, std::vector<GLuint> &elements);

std::unique_ptr<Model> new_quad(const std::function<void()> &setup_attribs);

#endif // MODEL_H
