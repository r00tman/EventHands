#include "metadatawriter.h"

MetadataWriter::MetadataWriter(const std::string &_fn, int _nComps) {
    fn = _fn;
    nComps = _nComps;

    file = fopen(fn.c_str(), "wb");
    fwrite(&nComps, sizeof(int), 1, file);
}

MetadataWriter::~MetadataWriter() {
    fclose(file);
}

void MetadataWriter::write_poses(const xt::xarray<double> &pose) {
    if (nComps != pose.size()) {
        throw std::runtime_error("writer: nComps doesn't match passed pose size: "+std::to_string(pose.size()));
    }
    fwrite_unlocked(pose.data(), sizeof(double), pose.size(), file);
}

void MetadataWriter::finalize_frame() {
    char magic[] = {4, 13};
    fwrite_unlocked(magic, sizeof(char), sizeof(magic)/sizeof(char), file);
}
