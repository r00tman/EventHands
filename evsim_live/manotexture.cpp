#include "manotexture.h"

#include <cstdio>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>

using namespace xt::placeholders;

xt::xarray<float> load_bin(const std::string &fn, int nRows, int nCols) {
    FILE *file = fopen(fn.c_str(), "rb");
    if (file == NULL) {
        fprintf(stderr, "failed to open .bin file: %s\n", fn.c_str());
        return xt::xarray<float>::from_shape({(unsigned long)nRows, (unsigned long)nCols});
    } else {
        xt::xarray<float> result = xt::zeros<float>({(unsigned long)nCols, (unsigned long)nRows});
//        const int blockSize = 32768/4;
//        const int totalSize = nCols*nRows;
//        int read = 0;
//        while (read < totalSize) {
//            int cnt = totalSize-read;
//            if (cnt > blockSize) {
//                cnt = blockSize;
//            }
//            fread_unlocked(result.data()+read, sizeof(float), cnt, file);
//            read += cnt;
//        }
        fread_unlocked(result.data(), sizeof(float), nCols*nRows, file);
//        xt::xarray<float> result = xt::zeros<float>({nRows, nCols});

//        for (int i = 0; i < nCols; ++i) {
//            for (int j = 0; j < nRows; ++j) {
//                fread_unlocked(&(result(j, i)), sizeof(float), 1, file);
//            }
//        }
        fclose(file);
        xt::xarray<float> result_transposed = xt::eval(xt::transpose(result));
        return result_transposed;
    }
}

ManoTexture::ManoTexture() : Texture("exampleTexture.png") {
    stddev = load_bin("std_dev_matrix.bin", 101, 101);
    eigen = load_bin("eigen_vector_matrix.bin", 3*1024*1024, 101);

    mean = load_bin("mean_data_vec.bin", 3*1024*1024, 1);
    param = xt::zeros<float>({101, 1});
}

void ManoTexture::change() {
    const int width = 1024, height = 1024;
    const float magnitude = 2.0f;
    param = xt::random::randn<float>({101, 1})*magnitude;

    xt::xarray<float> result;
    result = mean + xt::linalg::dot(eigen, xt::linalg::dot(stddev, param));
    result = xt::clip(result, 0.0f, 255.0f);

    xt::xarray<uint8_t> result_uchar = xt::cast<uint8_t>(result);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, result_uchar.data());
}
