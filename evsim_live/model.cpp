#include "model.h"

#include <iostream>
#include <fstream>
#include <glm/glm.hpp>

#include "log.h"

Model::Model(const std::vector<float> &_vertices, const std::vector<GLuint> &_elements, bool dynamic,
             const std::function<void()> &setup_attribs) {
    vertices = _vertices;
    elements = _elements;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);

    GLenum usage = dynamic ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW;
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*vertices.size(), vertices.data(), usage);

    setup_attribs();

    glGenBuffers(1, &ebo);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*elements.size(), elements.data(), usage);

    glBindVertexArray(0);
}

Model::~Model() {
    glDeleteBuffers(1, &ebo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

void Model::use() {
    glBindVertexArray(vao);
}

void Model::draw() {
    glDrawElements(GL_TRIANGLES, elements.size(), GL_UNSIGNED_INT, 0);
}

void Model::update_vertices() {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*vertices.size(), vertices.data());
}

void load_obj(const std::string &fn, std::vector<float> &vertices, std::vector<GLuint> &elements) {
    std::ifstream file(fn);
    while (file) {
        std::string token;
        file >> token;
        if (token == "v") {
            // vertex
            const float SCALE = 1;
            for (int i = 0; i < 3; ++i) {
                float x;
                file >> x;
                vertices.push_back(x*SCALE);
            }
            // normals
            vertices.push_back(0.0f);
            vertices.push_back(0.0f);
            vertices.push_back(0.0f);
        } else if (token == "f") {
            // face
            for (int i = 0; i < 3; ++i) {
                GLuint x;
                file >> x;
                elements.push_back(x-1);
            }
        } else if (token == "#" || token == "o" || token == "s") {
            // ignore the line
            std::getline(file, token);
        } else if (token.length() == 0) {
            // end of file
            break;
        } else {
            // unknown token
            std::cerr << token << std::endl;
            CHECK_NULL(NULL, "bad token");
        }
    }
    // flip y and z
    for (size_t i = 0; i < vertices.size(); i += 3) {
        std::swap(vertices[i+1], vertices[i+2]);
    }
    // compute normals
    auto vtx = [&](int i) { return glm::vec3(vertices[i+0], vertices[i+1], vertices[i+2]); };
    auto to_vtx = [&](int i, const glm::vec3 &x) { vertices[i+0] = x.x; vertices[i+1] = x.y; vertices[i+2] = x.z; };
    auto add_vtx = [&](int i, const glm::vec3 &x) { vertices[i+0] += x.x; vertices[i+1] += x.y; vertices[i+2] += x.z; };
    for (size_t i = 0; i < elements.size(); i += 3) {
        auto a = vtx(6*elements[i]);
        auto b = vtx(6*elements[i+1]);
        auto c = vtx(6*elements[i+2]);
        auto p = glm::cross(b-a, c-a);

        add_vtx(6*elements[i+0]+3, p);
        add_vtx(6*elements[i+1]+3, p);
        add_vtx(6*elements[i+2]+3, p);
    }
    // normalize normals
    for (size_t i = 3; i < vertices.size(); i += 6) {
        to_vtx(i, glm::normalize(vtx(i)));
    }
}

std::unique_ptr<Model> new_quad(const std::function<void()> &setup_attribs) {
    std::vector<float> vertices{
        -1.0, -1.0,
        -1.0, 1.0,
        1.0, 1.0,
        1.0, -1.0
    };
    std::vector<GLuint> elements{
        0, 1, 2,
        2, 3, 0
    };

    return std::unique_ptr<Model>(new Model(vertices, elements, false, setup_attribs));
}
