#include "manotypes.h"
#include <cfloat>
#include <xtensor-blas/xlinalg.hpp>

void print_shape(const xt::xarray<double> &arr) {
    const auto& s = arr.shape();
    std::copy(s.cbegin(), s.cend(), std::ostream_iterator<double>(std::cerr, " "));
    std::cerr << std::endl;
}

xt::xarray<double> rodrigues(xt::xarray<double> src) {
    xt::xarray<double> dst;

    if (src.shape(1) == 1 && src.shape(0) == 3) {
    } else {
        throw std::runtime_error("rodrigues: unsupported shape/transform");
    }

    xt::xarray<double> r = src;

    double theta = xt::linalg::norm(r);
    double itheta = theta ? 1./theta : 0.;

    r *= itheta;

    double cos_val = std::cos(theta);
    xt::xarray<double> cos_mat = cos_val * xt::eye(3);

    xt::xarray<double> cov_mat = (1-cos_val) * xt::linalg::dot(r, xt::transpose(r));

    double sin_val = std::sin(theta);
    xt::xarray<double> rot_mat{{0, -r(2,0), r(1,0)},
                               {r(2,0), 0, -r(0,0)},
                               {-r(1,0), r(0,0), 0}};

    dst = cos_mat + cov_mat + sin_val * rot_mat;
    return dst;
}

// normalize angle to [-pi, pi] range
double norm_angle(double ang) {
    const auto pi = std::acos(-1);
    while (ang < -pi) {
        ang += 2*pi;
    }
    while (ang > pi) {
        ang -= 2*pi;
    }
    assert(ang >= -pi);
    assert(ang <= pi);
    return ang;
}

xt::xarray<double> inv_rodrigues(xt::xarray<double> src) {
    xt::xarray<double> dst;

    if (src.shape(1) == 3 && src.shape(0) == 3) {
    } else {
        throw std::runtime_error("inv_rodrigues: unsupported shape/transform");
    }

    xt::xarray<double> diff = (src - xt::transpose(src))/2.;

    dst = {diff(2, 1)-diff(1, 2), diff(0, 2)-diff(2, 0), diff(1, 0)-diff(0, 1)};
    dst = dst / 2;

    double norm = xt::linalg::norm(dst);
    double inorm = norm ? 1./norm : 0.;
    dst = dst * inorm;
    dst = dst.reshape({3, 1});

    double sin_theta = xt::linalg::norm(diff)/std::sqrt(2.);

    // angle theta is either asin or pi-asin
    double theta1 = std::asin(sin_theta);
    double theta2 = std::acos(-1)-theta1;

    // normalize angles to [-pi,pi] range
    theta1 = norm_angle(theta1);
    theta2 = norm_angle(theta2);

    // we check which one it is by checking which one
    // gets the correct the forward transform values
    double err1 = xt::linalg::norm(rodrigues(dst*theta1)-src);
    double err2 = xt::linalg::norm(rodrigues(dst*theta2)-src);
    if (err1 < err2) {
        dst = dst*theta1;
        assert(err1 < 0.01);
    } else {
        dst = dst*theta2;
        assert(err2 < 0.01);
    }

//    std::cerr << 3 << " " << dst << " " << err1 << " " << err2 << std::endl;

    return dst;
}
