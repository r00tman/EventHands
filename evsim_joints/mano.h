#ifndef MANO_H
#define MANO_H

#include <vector>

#define GLEW_STATIC
#include <GL/glew.h>
#include <glm/glm.hpp>

#include <xtensor/xarray.hpp>

#include "manotypes.h"
#include "manogpu.h"

#include "manoanimator.h"

struct Mano {
private:
    bool loaded = false;
    smplData_t dd;

//    xt::xarray<double> poseCoeffs{0, 0, 0, 0, 0, 0, 0, 0, 0};
    xt::xarray<double> poseCoeffs;
//    xt::xarray<double> poseCoeffs = {
//        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // ncomps=12
//        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // ncomps=24
//        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // ncomps=36
//        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//        0, 0, 0, 0, 0, 0
//        };
//    xt::xarray<double> betas{0.00478618, 0.02003822, 0.00445073, 0.00528698, 0.01783764,
//                             0.02866372, 0.02793015, 0.01614217, 0.02167377, 0.00244703};
    xt::xarray<double> betas;
    bool poseChanged = true;
    bool betasChanged = true;

    xt::xarray<double> trans;

    glm::vec3 handPosition;
    glm::vec3 handRotation;

    bool needVShapedReupload = false;

    manoData_t manoData;
    bool isManoDataValid = false;

    result_t load_model(int nComps, bool flatHandMean, bool wantVPosed);
    void ready_arguments(bool wantVPosed);
    void lbs_verts_core();
    xt::xarray<double> global_rigid_transformation();
    void compute_normals(const std::vector<float> &vertices, std::vector<float> &normals,
                         const std::vector<GLuint> &elements);
    void compute_mano_and_normals(GLuint vbo, GLuint nbo, GLuint ebo);

public:
    Mano();
    ~Mano();

    void generate(std::vector<float> &vertices,
                  std::vector<float> &normals,
                  std::vector<GLuint> &elements,
                  std::vector<float> &weights,
                  std::vector<float> &texcoords,
                  std::vector<glm::mat4x3> &A);
    void generate(GLuint vbo, GLuint nbo, GLuint ebo, std::vector<glm::mat4x3> &A);
    void change_pose(bool amend);
    void change_betas();

    xt::xarray<double> get_pose();

    glm::vec3 get_trans();

    glm::vec3 get_hand_position();
    glm::vec3 get_hand_rotation();
    xt::xarray<double> get_joints();

    long double t = 0;
    long double dt = 1/60.L;

    const static int N_COMPS = 2*6;
//    const static int N_COMPS = 2*45;

    ManoAnimator *animator = NULL;
};

#endif // MANO_H
