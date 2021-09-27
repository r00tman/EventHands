#include "mano.h"

#include <functional>
#include <iostream>
#include <string>

#include <glm/gtc/type_ptr.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor-blas/xlapack.hpp>
#include <xtensor-blas/xblas.hpp>

#include "config.h"

#include "manoloader.h"
#include "manogpu.h"

using namespace xt::placeholders;


xt::xarray<double> posemap(const std::string &bsType, xt::xarray<double> x) {
    if (bsType != "lrotmin") {
        throw std::runtime_error("posemap: unknown posemapping");
    }
    x = x.reshape({-1, 3});
    x = xt::view(x, xt::range(1, _));
    xt::xarray<double> res = xt::zeros<double>({x.shape(0), 9ul});

    for (size_t i = 0; i < x.shape(0); ++i) {
        xt::xarray<double> inp = xt::view(x, i);
        inp = inp.reshape({3, 1});
        auto mat = rodrigues(inp);
        xt::xarray<double> sub = mat - xt::eye(3);
        sub = sub.reshape({9ul});
        xt::view(res, i, xt::all()) = sub;
    }

    return res.reshape({-1});
}

void Mano::ready_arguments(bool wantVPosed) {
    bool wantShapeModel = (dd.shapeDirs.size()!=1);
    int nPoseParms = dd.kintreeTable.shape(1)*3;

    if (dd.pose.size() == 1) {
        dd.pose = xt::zeros<double>({nPoseParms});
    }
    if (dd.shapeDirs.size() != 1 && dd.betas.size() == 1) {
        dd.betas = xt::zeros<double>({dd.shapeDirs.shape(2)});
    }

    if (wantShapeModel) {
        if (betasChanged) {
            dd.vShaped = xt::linalg::dot(dd.shapeDirs, dd.betas)+dd.vTemplate;
            needVShapedReupload = true;

            xt::xarray<double> inpx = xt::view(dd.vShaped, xt::all(), 0);
            auto Jtmpx = xt::linalg::dot(dd.Jregressor, inpx);
            xt::xarray<double> inpy = xt::view(dd.vShaped, xt::all(), 1);
            auto Jtmpy = xt::linalg::dot(dd.Jregressor, inpy);
            xt::xarray<double> inpz = xt::view(dd.vShaped, xt::all(), 2);
            auto Jtmpz = xt::linalg::dot(dd.Jregressor, inpz);
            dd.J = xt::transpose(xt::vstack(xt::xtuple(Jtmpx, Jtmpy, Jtmpz)));
        }

        if (betasChanged || poseChanged) {
            dd.posemapped = posemap(dd.bsType, dd.fullPose);
            if (wantVPosed) {
                dd.vPosed = dd.vShaped + xt::linalg::dot(dd.poseDirs, dd.posemapped);
            }
        }
    } else {
        throw std::runtime_error("mano: only shape model is supported");
        if (poseChanged) {
            dd.posemapped = posemap(dd.bsType, dd.fullPose);
            if (wantVPosed) {
                dd.vPosed = dd.vTemplate + xt::linalg::dot(dd.poseDirs, dd.posemapped);
            }
        }
    }
}

xt::xarray<double> Mano::global_rigid_transformation() {
    xt::xarray<double> result;
    std::vector<xt::xarray<double> > results;

    xt::xarray<double> pose = dd.fullPose.reshape({-1, 3});
    std::map<int64_t, int64_t> idToCol;
    for (size_t i = 0; i < dd.kintreeTable.shape(1); ++i) {
        idToCol[dd.kintreeTable(1, i)] = i;
    }
    std::map<int64_t, int64_t> parent;
    for (size_t i = 1; i < dd.kintreeTable.shape(1); ++i) {
        parent[i] = idToCol[dd.kintreeTable(0, i)];
    }

    xt::xarray<double> zeros = {{0.0, 0.0, 0.0, 1.0}};
    auto with_zeros = [&](auto x) {
        return xt::vstack(xt::xtuple(x, zeros));
    };
    results.emplace_back(with_zeros(xt::hstack(xt::xtuple(
        rodrigues(xt::view(pose, 0, xt::all(), xt::newaxis())),
        xt::reshape_view(xt::view(dd.J, 0, xt::all()), {3, 1})
    ))));

    for (size_t i = 1; i < dd.kintreeTable.shape(1); ++i) {
        auto prnt = results[parent[i]];
        auto cpose = xt::view(pose, i, xt::all(), xt::newaxis());
        auto rod = rodrigues(cpose);
        auto cJ = xt::view(dd.J, i, xt::all());
        auto pJ = xt::view(dd.J, parent[i], xt::all());

        results.emplace_back(xt::linalg::dot(prnt,
               with_zeros(xt::hstack(xt::xtuple(
                   rod,
                   xt::reshape_view(
                       cJ - pJ,
                       {3, 1}))
               ))));
    }

    auto pack = [](auto x) {
        return xt::hstack(xt::xtuple(xt::zeros<double>({4, 3}), x.reshape({4, 1})));
    };

    dd.Jtr = xt::zeros<double>({results.size(), 3ul});
    for (size_t i = 0; i < results.size(); ++i) {
        xt::xarray<double> translation = xt::view(results[i], xt::range(_, 3), 3);
        // if it's right hand joint, extract position and rotation
        if (i == 21) {
//        if (i == int(21+t/5)) {
//            std::cerr << int(21+t/5) << ' ' << results.size() << std::endl;
//            print_shape(results[i]);
//            xt::xarray<double> t = xt::transpose(results[i]);
            xt::xarray<double> transform = results[i];
//            t = {1, 1, 1};
//            t = xt::reshape_view(t, {3, 1});
//            std::cerr << transform << std::endl;
//            std::cerr << inv_rodrigues(rodrigues(t)) << std::endl;
//            t = rodrigues(t);
//            std::cerr << p << std::endl;
//            xt::view(t, xt::range(_, 3), xt::range(_, 3)) = rodrigues(inv_rodrigues(
//                                  xt::view(t, xt::range(_, 3), xt::range(_, 3))));
            xt::xarray<double> rotation =
                    inv_rodrigues(xt::view(transform, xt::range(_, 3), xt::range(_, 3)));
            handRotation = glm::make_vec3(&rotation(0));
            handPosition = glm::make_vec3(&translation(0));
//            std::cerr << t << std::endl;
//            t = xt::transpose(t);
//            transform = glm::make_mat4(&t.at(0));
//            xt::xarray<double> p = xt::view(results[i], xt::range(_, 3), 3);
        }
        xt::view(dd.Jtr, i, xt::range(_, _)) = translation;
    }

    std::vector<xt::xarray<double> > results2;
    for (size_t i = 0; i < results.size(); ++i) {
        xt::xarray<double> jv = xt::view(dd.J, i, xt::all());
        auto what = xt::concatenate(xt::xtuple(jv,
                                               xt::xarray<double>{0}));
        results2.push_back(results[i] - pack(xt::linalg::dot(results[i], what)));
    }
    results = results2;
    result = xt::zeros<double>({results[0].shape(0), results[0].shape(1), results.size()});
    for (size_t i = 0; i < results.size(); ++i) {
        xt::view(result, xt::all(), xt::all(), i) = results[i];
    }

    return result; // needs to be result, resultsGlobal
}

void Mano::lbs_verts_core() {
    assert(dd.bsStyle == "lbs");
    dd.A = global_rigid_transformation();
}

result_t Mano::load_model(int nComps, bool flatHandMean, bool wantVPosed) {
    if (!loaded) {
        dd = load_data();
//        animator = new ManoFileAnimator("gt.txt");
//        animator = new ManoFileAnimator("pr.txt");
        if (PLAYBACK.length()) {
            if (FIFO) {
                animator = new ManoFifoAnimator(PLAYBACK, nComps);
            } else {
                animator = new ManoFileAnimator(PLAYBACK, nComps);
            }
        } else {
            animator = new ManoBezierAnimator(nComps);
        }
        loaded = true;
    }
    if (poseChanged) {
        const int bodyPoseDofs = 66;

        auto handsComponentsL = xt::view(dd.handsComponentsL, xt::range(_, nComps/2));
        auto handsMeanL = flatHandMean ? xt::zeros<double>({dd.handsComponentsL.shape(1)}) :
                                         dd.handsMeanL;

        auto handsComponentsR = xt::view(dd.handsComponentsR, xt::range(_, nComps/2));
        auto handsMeanR = flatHandMean ? xt::zeros<double>({dd.handsComponentsR.shape(1)}) :
                                         dd.handsMeanR;

        auto selectedComponents = xt::vstack(xt::xtuple(
                                                 xt::hstack(xt::xtuple(handsComponentsL, xt::zeros_like(handsComponentsL))),
                                                 xt::hstack(xt::xtuple(xt::zeros_like(handsComponentsR), handsComponentsR))
                                                 ));
        auto handsMean = xt::concatenate(xt::xtuple(handsMeanL, handsMeanR));
        auto handPose = xt::view(poseCoeffs, xt::range(bodyPoseDofs, bodyPoseDofs+nComps));
//        std::cerr << bodyPoseDofs << " " << bodyPoseDofs + nComps << std::endl;
//        std::cerr << xt::adapt(handPose.shape()) << ' ' << xt::adapt(selectedComponents.shape()) << std::endl;
        auto fullHandPose = xt::linalg::dot(handPose, selectedComponents);

        dd.fullPose = xt::concatenate(xt::xtuple(xt::view(poseCoeffs, xt::range(_, bodyPoseDofs)), (handsMean + fullHandPose)));
//        const int rot = 3;

//        auto handsMean = flatHandMean ? xt::zeros<double>({dd.handsComponents.shape(1)}) :
//                                        dd.handsMean;

//        auto handsCoeffs = xt::view(dd.handsCoeffs, xt::range(_, nComps));

//        auto selectedComponents = xt::vstack(xt::xtuple(xt::view(dd.handsComponents, xt::range(_, nComps))));

//        auto fullHandPose = xt::linalg::dot(xt::view(poseCoeffs, xt::range(rot, rot+nComps)), selectedComponents);

//        dd.fullPose = xt::concatenate(xt::xtuple(xt::view(poseCoeffs, xt::range(_, rot)), (handsMean + fullHandPose)));
//        dd.pose = poseCoeffs;
    }

    if (betasChanged) {
        dd.betas = betas;
    }

    ready_arguments(wantVPosed);
    lbs_verts_core();

    result_t result;
    if (wantVPosed) {
        result.f = dd.f;
        result.weights = xt::zeros<double>({dd.duplicateMap.size(), dd.weights.shape(1)});
        for (int i = 0; i < dd.duplicateMap.size(); ++i) {
            xt::view(result.weights, i, xt::all()) = xt::view(dd.weights, dd.duplicateMap(i), xt::all());
        }
        result.res = xt::zeros<double>({dd.duplicateMap.size()*3UL,});
        for (int i = 0; i < dd.duplicateMap.size(); ++i) {
            result.res(i*3+0) = dd.vPosed(dd.duplicateMap(i)*3+0);
            result.res(i*3+1) = dd.vPosed(dd.duplicateMap(i)*3+1);
            result.res(i*3+2) = dd.vPosed(dd.duplicateMap(i)*3+2);
        }
        // todo: implement vertice duplication in the gpu version too
    }

    poseChanged = false;
    betasChanged = false;

    return result;

}

void Mano::compute_normals(const std::vector<float> &vertices, std::vector<float> &normals,
                           const std::vector<GLuint> &elements) {
    auto vtx = [&](int i) {
        return glm::vec3(vertices[i+0], vertices[i+1], vertices[i+2]);
    };
    auto norm = [&](int i) {
        return glm::vec3(normals[i+0], normals[i+1], normals[i+2]);
    };
    auto to_norm = [&](int i, const glm::vec3 &x) {
        normals[i+0] = x.x;
        normals[i+1] = x.y;
        normals[i+2] = x.z;
    };
    auto add_norm = [&](int i, const glm::vec3 &x) {
        normals[i+0] += x.x;
        normals[i+1] += x.y;
        normals[i+2] += x.z;
    };
//    ::compute_normals(vertices.data(), elements.data(), vertices.size()/6, elements.size()/3);
    for (size_t i = 0; i < normals.size(); i += 3) {
        to_norm(i, glm::vec3(0));
    }
    // for each triangle (3 vertices)
    for (size_t i = 0; i < elements.size(); i += 3) {
        // a vertex contains 3 positions + 3 normals = 6 values
        auto a = vtx(3*elements[i+0]);
        auto b = vtx(3*elements[i+1]);
        auto c = vtx(3*elements[i+2]);
        // a weighted normal vector to the triangle
        auto p = glm::cross(b-a, c-a);

        add_norm(3*elements[i+0], p);
        add_norm(3*elements[i+1], p);
        add_norm(3*elements[i+2], p);
    }
    // normalize normals
    for (size_t i = 0; i < normals.size(); i += 3) {
        to_norm(i, glm::normalize(norm(i)));
    }
}

void Mano::compute_mano_and_normals(GLuint vbo, GLuint nbo, GLuint ebo) {
    if (!isManoDataValid) {
        xt::xarray<float> vShaped = xt::cast<float>(dd.vShaped);
        xt::xarray<float> poseDirs = xt::cast<float>(dd.poseDirs);
        init_mano_gpu(&manoData, vbo, nbo, ebo, vShaped.data(), poseDirs.data(), dd.vTemplate.size()/3, dd.f.size()/3);
        isManoDataValid = true;
        needVShapedReupload = false;
    }
    if (needVShapedReupload) {
        xt::xarray<float> vShaped = xt::cast<float>(dd.vShaped);
        update_mano_vshaped(&manoData, vShaped.data());
        needVShapedReupload = false;
    }
    xt::xarray<float> posemapped = xt::cast<float>(dd.posemapped);
    manoData.hostPosemapped = posemapped.data();
    compute_mano_and_normals_vbo(&manoData);
}

Mano::Mano() {
    poseCoeffs = xt::zeros<double>({66UL+N_COMPS,});
    betas = xt::zeros<double>({10UL,});
}

Mano::~Mano() {
    if (isManoDataValid) {
        destroy_mano_gpu(&manoData);
    }
    if (animator != NULL) {
        delete animator;
        animator = NULL;
    }
}

void Mano::generate(std::vector<float> &vertices,
                    std::vector<float> &normals,
                    std::vector<GLuint> &elements,
                    std::vector<float> &weights,
                    std::vector<float> &texcoords,
                    std::vector<glm::mat4x3> &A) {
    vertices.clear();
    normals.clear();
    elements.clear();
    weights.clear();
    texcoords.clear();
    A.clear();


    result_t result = load_model(N_COMPS, false, true);

    // scale should be 1, because MANO model is yet to be
    // transformed and skinned in the vertex shader
    const float SCALE = 1;

    // positions
    for (size_t i = 0; i < result.res.size(); i += 3) {
        vertices.push_back(result.res(i+0)*SCALE);
        vertices.push_back(result.res(i+1)*SCALE);
        vertices.push_back(result.res(i+2)*SCALE);
    }

    // normals
    for (size_t i = 0; i < result.res.size(); i += 3) {
        normals.push_back(0);
        normals.push_back(0);
        normals.push_back(0);
    }

    // weights & indices are static. should we repopulate that every time?
    // it's barely noticeable in terms of time, though.
    for (size_t i = 0; i < result.f.size(); ++i) {
        elements.push_back(result.f(i));
    }

    auto w = result.weights.reshape({-1});
    for (size_t i = 0; i < w.size(); ++i) {
        weights.push_back(w(i));
    }

    for (size_t i = 0; i < dd.texCoords.size(); ++i) {
        texcoords.push_back(dd.texCoords[i].x);
        texcoords.push_back(dd.texCoords[i].y);
    }

    for (size_t i = 0; i < dd.A.shape(2); ++i) {
        float mat[4*3];
        // we take just first three components from A, so mat4x3 instead of mat4x4
        // and btw we know that glm is column-major and xtensor is row-major
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                mat[c*3+r] = dd.A(r, c, i);
            }
        }
        glm::mat4x3 r = glm::make_mat4x3(mat);
        A.emplace_back(r);
    }

    // todo: compute normals before duplication
    compute_normals(vertices, normals, elements);

    // average normals on the seams
    std::vector<float> newNormals(normals.size());

    for (size_t dest = 0; dest < dd.duplicateMap.size(); ++dest) {
        int src = dd.duplicateMap(dest);
        newNormals[dest*3+0] += normals[src*3+0];
        newNormals[dest*3+1] += normals[src*3+1];
        newNormals[dest*3+2] += normals[src*3+2];
    }

    for (size_t dest = 0; dest < dd.duplicateMap.size(); ++dest) {
        glm::vec3 out = glm::normalize(glm::vec3(newNormals[dest*3+0], newNormals[dest*3+1], newNormals[dest*3+2]));
        newNormals[dest*3+0] = out.x;
        newNormals[dest*3+1] = out.y;
        newNormals[dest*3+2] = out.z;
    }

    normals = newNormals;
}

void Mano::generate(GLuint vbo, GLuint nbo, GLuint ebo, std::vector<glm::mat4x3> &A) {
    A.clear();

    // last false: don't want posed model, we'll do it on gpu
    result_t result = load_model(N_COMPS, false, false);

    for (size_t i = 0; i < dd.A.shape(2); ++i) {
        float mat[4*3];
        // we take just first three components from A, so mat4x3 instead of mat4x4
        // and btw we know that glm is column-major and xtensor is row-major
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                mat[c*3+r] = dd.A(r, c, i);
            }
        }
        glm::mat4x3 r = glm::make_mat4x3(mat);
        A.emplace_back(r);
    }

//    glBindBuffer(GL_ARRAY_BUFFER, vbo);
//    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*vertices.size(), vertices.data());
    compute_mano_and_normals(vbo, nbo, ebo);
}

void Mano::change_pose(bool amend) {
    animator->set_dt(dt);
    t = animator->get_t();
    animator->change_pose(poseCoeffs, trans, amend);
//    poseCoeffs = xt::zeros<double>({78,});
//    trans = xt::zeros<double>({3,});

    poseChanged = true;
//    poseChanged = false;
}

void Mano::change_betas() {
    betas = xt::random::rand<double>({10ul}, -2, 2);
    betasChanged = true;
}

xt::xarray<double> Mano::get_pose() {
//    return poseCoeffs;
    xt::xarray<double> manoArray = xt::view(poseCoeffs, xt::range(66+N_COMPS-N_COMPS/2, 66+N_COMPS));

    glm::vec3 pos = get_hand_position();
    xt::xarray<double> posArray{pos.x, pos.y, pos.z};

    glm::vec3 rot = get_hand_rotation();
    xt::xarray<double> rotArray{rot.x, rot.y, rot.z};

    return xt::concatenate(xt::xtuple(manoArray, posArray, rotArray));
}

glm::vec3 Mano::get_trans() {
    // global translation
    return glm::vec3(trans(0), trans(1), trans(2));
}

glm::vec3 Mano::get_hand_position() {
//    const int idx = t/2+21;
//    const int idx = 21;
//    const int idx = t/2+37;
//    std::cerr << idx << std::endl;
//    return glm::vec3(dd.Jtr(idx, 0)+trans(0), dd.Jtr(idx, 1)+trans(1), dd.Jtr(idx, 2)+trans(2));
    return handPosition+get_trans();
}

glm::vec3 Mano::get_hand_rotation() {
    return handRotation;
}

xt::xarray<double> Mano::get_joints() {
    return dd.Jtr;
}
