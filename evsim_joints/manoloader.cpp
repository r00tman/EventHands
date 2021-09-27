#include "manoloader.h"

#include <vector>
#include <map>

#include <xtensor-io/xnpz.hpp>
#include <glm/glm.hpp>

std::vector<ulong> load_doubled_vertices() {
    std::map<int, int> res_map;
    // do I use the regular one or the flipped one?
    std::ifstream file("duplicate_map_flipped.txt");

    int src, dest;
    std::string arrow;
    // preserve original vertices
    for (int i = 0; i < 778; ++i) {
        res_map[i] = i;
    }
    while (file >> src >> arrow >> dest >> std::ws) {
        assert(arrow == "->");
        if (dest != 0) {
            // check that this is the first time we see
            // that destination vertex, i.e., we're not
            // overwriting another vertex, or better said,
            // there are no duplicate destination vertices
            assert(res_map.find(dest) == res_map.end());
            res_map[dest] = src;
        }
    }
    std::vector<ulong> res;
    // check that it all starts with the first vertex
    assert(res_map.begin()->first == 0);
    int last = res_map.begin()->first-1;
    for (const auto &x : res_map) {
        // check validity of the source vertex
        // (e.g., we're not duplicating a duplicate)
        assert(x.second >= 0);
        assert(x.second < 778 || x.first == x.second);
        // check monotonicity and denseness of the duplicate verts
        // i.e., no idx is missed or duplicated
        assert(x.first == last + 1);
        last = x.first;
        res.push_back(x.second);
    }
    return res;
}

std::vector<glm::vec2> load_tex_coords() {
//    std::ifstream file("uv_coords_flipped.txt");
    std::ifstream file("new_uvs.txt");

    std::vector<glm::vec2> res;

    glm::vec2 uv;
    while (file >> uv.x >> uv.y >> std::ws) {
        res.emplace_back(uv);
    }

    return res;
}

xt::xarray<uint32_t> load_faces() {
    std::ifstream file("faces_dup_flipped.txt");
    std::vector<glm::vec3> res;

    glm::u32vec3 face;
    while (file >> face.x >> face.y >> face.z >> std::ws) {
        res.emplace_back(face);
    }

    xt::xarray<uint32_t>::shape_type shape = {res.size(), 3};
    xt::xarray<uint32_t> res_xt(shape);

    for (int i = 0; i < res.size(); ++i) {
        res_xt(i, 0) = res[i].x;
        res_xt(i, 1) = res[i].y;
        res_xt(i, 2) = res[i].z;
    }

    res_xt = xt::flatten(res_xt);

    return res_xt;
}

smplData_t load_data() {
    smplData_t smpl;
    auto file = xt::load_npz("smpl.npz");
    smpl.J = file["J"].cast<double>();
    smpl.Jregressor = file["J_regressor"].cast<double>();
    smpl.f = file["f"].cast<uint32_t>();
    smpl.handsCoeffsL = file["hands_coeffsl"].cast<double>();
    smpl.handsComponentsL = file["hands_componentsl"].cast<double>();
    smpl.handsMeanL = file["hands_meanl"].cast<double>();
    smpl.handsCoeffsR = file["hands_coeffsr"].cast<double>();
    smpl.handsComponentsR = file["hands_componentsr"].cast<double>();
    smpl.handsMeanR = file["hands_meanr"].cast<double>();
    smpl.kintreeTable = file["kintree_table"].cast<int64_t>();
    smpl.poseDirs = file["posedirs"].cast<double>();
    smpl.shapeDirs = file["shapedirs"].cast<double>();
    smpl.vTemplate = file["v_template"].cast<double>();
    smpl.weights = file["weights"].cast<double>();

    smpl.bsStyle = "lbs";
    smpl.bsType = "lrotmin";

//    auto duplicateMap = load_doubled_vertices();
    std::vector<ulong> duplicateMap;
    for (int i = 0; i < smpl.vTemplate.size()/3; ++i) {
        duplicateMap.push_back(i);
    }

    smpl.duplicateMap = xt::adapt(duplicateMap);
    smpl.texCoords = load_tex_coords();
//    smpl.texCoords.clear();
//    for (int i = 0; i < smpl.duplicateMap.size(); ++i) {
//        smpl.texCoords.emplace_back(glm::vec2(0.0, 0.0));
//    }
    // check that every texcoord is one-to-one corresponding to the vertices
    assert(smpl.duplicateMap.size() == smpl.texCoords.size());

//    smpl.f = load_faces();

    return smpl;
}
