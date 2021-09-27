#ifndef MANOTYPES_H
#define MANOTYPES_H

#include <vector>
#include <xtensor/xarray.hpp>
#include <glm/glm.hpp>

struct smplData_t {
    xt::xarray<double> pose, betas;
    xt::xarray<double> poseDirs, shapeDirs, Jregressor, J, fullPose;
    xt::xarray<double> posemapped;
    xt::xarray<int64_t> kintreeTable;
    xt::xarray<double> vTemplate, vShaped, vPosed;
    xt::xarray<uint32_t> f;

    xt::xarray<double> handsComponentsL, handsCoeffsL, handsMeanL;
    xt::xarray<double> handsComponentsR, handsCoeffsR, handsMeanR;
    xt::xarray<double> weights, A, Jtr;
    std::string bsType;
    std::string bsStyle;

    xt::xarray<ulong> duplicateMap;
    std::vector<glm::vec2> texCoords;
};

struct result_t {
    xt::xarray<double> res, weights;
    xt::xarray<int> f;
};

void print_shape(const xt::xarray<double> &arr);
xt::xarray<double> rodrigues(xt::xarray<double> src);
xt::xarray<double> inv_rodrigues(xt::xarray<double> src);

#endif // MANOTYPES_H
