#ifndef METADATAWRITER_H
#define METADATAWRITER_H

#include <cstdio>
#include <string>
#include <xtensor/xarray.hpp>

struct MetadataWriter {
private:
    std::string fn;
    int nComps;

    FILE *file;

public:
    MetadataWriter(const std::string &_fn, int _nComps);
    ~MetadataWriter();

    void write_poses(const xt::xarray<double> &pose);
    void finalize_frame();
};

#endif // METADATAWRITER_H
