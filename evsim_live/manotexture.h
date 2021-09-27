#ifndef MANOTEXTURE_H
#define MANOTEXTURE_H

#include "texture.h"
#include <xtensor/xarray.hpp>

struct ManoTexture : public Texture {
    xt::xarray<float> mean, eigen, stddev, param;

    ManoTexture();

    void change();
};

#endif // MANOTEXTURE_H
