#ifndef MANOANIMATOR_H
#define MANOANIMATOR_H
#include <vector>
#include <string>
#include <xtensor/xarray.hpp>
#include <glm/glm.hpp>


struct ManoAnimator {
public:
    int NCOMP;

    ManoAnimator(int ncomp);
    virtual ~ManoAnimator() {}

    virtual void change_pose(xt::xarray<double> &pose, xt::xarray<double> &trans, bool amend) = 0;
    virtual void set_dt(long double dt) = 0;
    virtual long double get_t() = 0;
};

struct ManoBezierAnimator : public ManoAnimator {
private:
    long double tlocal = 0;
    long double t = 0;
    long double dt = 1/60.f;

    xt::xarray<double> savedPoseCoeffs;
    xt::xarray<double> savedTrans;

    xt::xarray<double> curPose;
    xt::xarray<double> nextPose;
    xt::xarray<double> nextNextPose;

    xt::xarray<double> curTrans;
    xt::xarray<double> nextTrans = {0, 0, 0};
    xt::xarray<double> nextNextTrans = {0, 0, 0};

public:
    ManoBezierAnimator(int ncomp);

    virtual void change_pose(xt::xarray<double> &pose, xt::xarray<double> &trans, bool amend);
    virtual void set_dt(long double _dt);
    virtual long double get_t();
};

struct ManoFileAnimator : public ManoAnimator {
private:
    long double t = 0;
    long double file_fps = 1000;
    long double display_dt = 1/60.f;
    int previdx = 0;
    std::vector<xt::xarray<double> > poses, translations, rotations;

public:
    ManoFileAnimator(const std::string &fn, int ncomp);

    virtual void change_pose(xt::xarray<double> &pose, xt::xarray<double> &trans, bool amend);
    virtual void set_dt(long double dt);
    virtual long double get_t();

    glm::vec3 mytrans, myrot;
};

struct ManoFifoAnimator : public ManoAnimator {
private:
    long double t = 0;
    long double file_fps = 60;
    long double display_dt = 1/60.f;
    int previdx = 0;
    int fifo;
    xt::xarray<double> rpose, rtrans, rrot;

public:
    ManoFifoAnimator(const std::string &fn, int ncomp);

    virtual void change_pose(xt::xarray<double> &pose, xt::xarray<double> &trans, bool amend);
    virtual void set_dt(long double dt);
    virtual long double get_t();

    glm::vec3 mytrans, myrot;
};
#endif // MANOANIMATOR_H
