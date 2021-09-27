#include "manoanimator.h"
#include <fstream>
#include <sstream>
#include <xtensor/xrandom.hpp>
#include <xtensor/xsort.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <fcntl.h>
#include <unistd.h>

#include "config.h"

using namespace xt::placeholders;

ManoAnimator::ManoAnimator(int ncomp) : NCOMP(ncomp) {

}

xt::xarray<double> generate_pose(int NCOMP, double random_coef=0.01) {
    xt::xarray<double> pose;

    pose = xt::random::rand<double>({66+NCOMP}, -2., 2.);
//    pose = xt::random::randn<double>({66+NCOMP}, 0 , 1);
//    pose = xt::random::randn<double>({78ul}, 0, 1.);
//    xt::view(pose, xt::range(78-6, _)) = xt::random::randn<double>({6ul}, 0, 1.);

//    // filter just params that are related to arms
//    const float coeff = 0.1;
//    const int beg1 = 3*13;
//    const int end1 = beg1+6;
//    const int beg2 = 3*16;
//    const int end2 = beg2+18;
//    xt::view(pose, xt::range(_, beg1)) *= 0.0;
//    xt::view(pose, xt::range(beg1, end1)) *= coeff;
//    xt::view(pose, xt::range(end1, beg2)) *= 0.0;
//    xt::view(pose, xt::range(beg2, end2)) *= coeff;
//    xt::view(pose, xt::range(end2, 78-12)) *= 0.0;

    // no movement at all
//    xt::view(pose, xt::range(_, 78)) *= 0.0;
//    const int beg = 13*3+3;
//    const int end = 13*3+4;

//    xt::view(pose, xt::range(_, beg)) *= 0.0;
//    xt::view(pose, xt::range(beg, end)) *= 1.0;
//    xt::view(pose, xt::range(end, 78-6)) *= 0.0;

    xt::view(pose, xt::range(_, 66+NCOMP-NCOMP/2)) *= random_coef;

//    pose.at(0) *= 100;
//    pose.at(1) *= 100;
//    pose.at(2) *= 100;

    pose.at(16*3+9) *= 100;
    pose.at(16*3+11) *= 10;

    pose.at(16*3+10) *= 40;

    pose.at(16*3+15) *= 40;
    pose.at(16*3+16) *= 40;
    pose.at(16*3+17) *= 40;

    pose.at(13*3+3) += 0.0;
    pose.at(13*3+4) += 0.0;
    pose.at(13*3+5) += 0.2;

    pose.at(16*3+3) += 0.0;
    pose.at(16*3+4) += 0.0;
    pose.at(16*3+5) += 0.1;

    if (pose.at(16*3+9) > 0) {
        pose.at(16*3+9) += -1.4;
    } else {
        pose.at(16*3+9) += 1.4;
    }
    pose.at(16*3+10) += 0.0;
    pose.at(16*3+11) += 0.5;

    pose.at(16*3+15) += 0.0;
    pose.at(16*3+16) += 0.0;
    pose.at(16*3+17) += 0.0;


//    int idx = 17;
//    pose.at(16*3+idx) = 5*(rand()*2.0/RAND_MAX-1.0);
//    pose.at(16*3+idx) = 3;

    return pose;
}

ManoBezierAnimator::ManoBezierAnimator(int ncomp) : ManoAnimator(ncomp) {
    savedPoseCoeffs = generate_pose(NCOMP);
    nextPose = savedPoseCoeffs;
    nextNextPose = savedPoseCoeffs;

    savedTrans = xt::zeros<double>({3,});
}

void ManoBezierAnimator::change_pose(xt::xarray<double> &poseCoeffs, xt::xarray<double> &trans, bool amend) {
    // a pose target is changed every PERIOD seconds
    const long double PERIOD = 60.0L/60.0L;
    // blending speed
    const long double GAMMA = 0.02L;

    // fmod(0-dt, PERIOD)=-dt<0, so shift it by +PERIOD to make it non-negative
    if (fmodl(t, PERIOD) < fmodl(t-dt+PERIOD, PERIOD) || amend) {
        curPose = savedPoseCoeffs;
        nextPose = generate_pose(NCOMP);
        nextNextPose = generate_pose(NCOMP);

        curTrans = savedTrans;
        nextTrans = xt::random::rand<double>({3ul}, -0.3, 0.3);
        nextTrans(2) *= 0.3;
        nextNextTrans = xt::random::rand<double>({3ul}, -0.3, 0.3);
        nextNextTrans(2) *= 0.3;

        tlocal = 0;
    }
    if (!amend) {
        t += dt;
        tlocal += dt;
    }
    long double blend = tlocal/PERIOD;
//    blend /= 2;
    {
        xt::xarray<double> p1 = curPose+(nextPose-curPose)*blend;
        xt::xarray<double> p2 = nextPose+(nextNextPose-nextPose)*(blend);
        xt::xarray<double> p = p2*blend+p1*(1-blend);
        poseCoeffs = p;
    }
    {
        xt::xarray<double> p1 = curTrans+(nextTrans-curTrans)*blend;
        xt::xarray<double> p2 = nextTrans+(nextNextTrans-nextTrans)*(blend);
//        blend = 0.5;
        xt::xarray<double> p = p2*blend+p1*(1-blend);
        trans = p;
    }
//    trans *= 0;
//    poseCoeffs.at(13*3+3) = 0.4;
//    poseCoeffs.at(13*3+3) = -2.0;
//    poseCoeffs.at(13*3+3) *= 2;

    savedPoseCoeffs = poseCoeffs;
    savedTrans = trans;
//    // original math worked in terms of frames, i.e.,
//    // #frames since change=tlocal*60 fps
//    float blend = (1-exp(-tlocal*60.*GAMMA/(1-GAMMA)));
//    poseCoeffs = (nextTarget-srcPose)*blend+srcPose;
//    trans = (nextTrans-srcTrans)*blend+srcTrans;
////    poseCoeffs = poseCoeffs * (1-GAMMA) + GAMMA * nextTarget;
////    trans = xt::zeros<double>({3,});
////    trans(0) = sin(1.1*t);
////    trans(1) = cos(t+0.1);
////    trans(2) = sin(1.3*t+2);
////    trans = trans * 0.5;

}

void ManoBezierAnimator::set_dt(long double _dt) {
    dt = _dt;
}

long double ManoBezierAnimator::get_t() {
    return t;
}

ManoFileAnimator::ManoFileAnimator(const std::string &fn, int ncomp) : ManoAnimator(ncomp) {
    std::ifstream f(fn);

    std::cerr << "loading replay" << std::endl;
    while (f) {
        std::string line;
        std::getline(f, line);
        std::istringstream reader(line);
        xt::xarray<double> pose = xt::zeros<double>({NCOMP/2,});
        xt::xarray<double> trans = xt::zeros<double>({3,});
        xt::xarray<double> rot = xt::zeros<double>({3,});

        for (int i = 0; i < pose.size(); ++i) {
            reader >> pose.at(i);
        }
        for (int i = 0; i < 3; ++i) {
            reader >> trans.at(i);
        }
        for (int i = 0; i < 3; ++i) {
            reader >> rot.at(i);
        }
        poses.emplace_back(pose);
        translations.emplace_back(trans);
        rotations.emplace_back(rot);
    }
    std::cerr << "loaded replay" << std::endl;
}

void ManoFileAnimator::change_pose(xt::xarray<double> &pose, xt::xarray<double> &trans, bool amend) {
    assert(poses.size() > 0);
    assert(!amend);
    int idx = floor(fmod(t*file_fps, poses.size()));
    if (idx < previdx) {
        previdx = 0;
        if (VIDEO_RENDER) {
            fflush_unlocked(stdout);
            exit(0);
        }
    }
    previdx = idx;
    std::cerr << "requested frame " << idx << ' ' << t << ' ' << file_fps << ' ' << poses.size() << std::endl;
//    xt::xarray<double> npose = xt::zeros<double>({idx-previdx+1, 6});
//    xt::xarray<double> ntrans = xt::zeros<double>({idx-previdx+1, 3});
//    int cnt = 0;
//    for (; previdx <= idx; ++previdx) {
//        xt::view(npose, cnt, xt::range(_, 6)) = poses[idx];
//        xt::view(ntrans, cnt, xt::range(_, 3)) = translations[idx];
//        cnt++;
//    }
//    npose = xt::median(npose, 0);
//    ntrans = xt::median(ntrans, 0);

//    xt::xarray<double> npose = xt::zeros<double>({6,});
    xt::xarray<double> ntrans = xt::zeros<double>({3,});
//    int cnt = 0;
//    for (; previdx <= idx; ++previdx) {
//        npose += poses[idx];
//        ntrans += translations[idx];
//        cnt++;
//    }
//    npose /= cnt;
//    ntrans /= cnt;
    pose = generate_pose(NCOMP, 0.0);
    xt::view(pose, xt::range(66+NCOMP-NCOMP/2, 66+NCOMP)) = poses[idx];
//    trans = translations[idx];
    trans = ntrans;
//    rot = rotations[idx];
    mytrans = glm::make_vec3(&translations[idx](0));
    myrot = glm::make_vec3(&rotations[idx](0));
//    xt::view(pose, xt::range(78-6, 78)) = npose;
//    trans = ntrans;
    t += display_dt;
}

void ManoFileAnimator::set_dt(long double dt) {
    display_dt = dt;
}

long double ManoFileAnimator::get_t() {
    return t;
}

ManoFifoAnimator::ManoFifoAnimator(const std::string &fn, int ncomp) : ManoAnimator(ncomp) {
    fifo = open(fn.c_str(), O_RDONLY);
    fcntl(fifo, F_SETPIPE_SZ, 512);

    std::cerr << "starting livestream" << std::endl;
    rpose = xt::zeros<double>({6,});
    rtrans = xt::zeros<double>({3,});
    rrot = xt::zeros<double>({3,});
}

void ManoFifoAnimator::change_pose(xt::xarray<double> &pose, xt::xarray<double> &trans, bool amend) {
    assert(!amend);
    int idx = floor(t*file_fps);
//    int idx = previdx + 1;
    if (idx < previdx) {
        previdx = 0;
    }
    std::cerr << "requested frame " << idx << ' ' << t << ' ' << file_fps << std::endl;

    while (previdx < idx)
    {
        read(fifo, &rpose(0), sizeof(double)*6);
        read(fifo, &rtrans(0), sizeof(double)*3);
        read(fifo, &rrot(0), sizeof(double)*3);
        previdx++;
    }

    pose = generate_pose(NCOMP, 0.0);
    xt::view(pose, xt::range(66+NCOMP-NCOMP/2, 66+NCOMP)) = rpose;
    trans = xt::zeros<double>({3,});
    mytrans = glm::make_vec3(&rtrans(0));
    myrot = glm::make_vec3(&rrot(0));
    t += display_dt;
}

void ManoFifoAnimator::set_dt(long double dt) {
    display_dt = dt;
}

long double ManoFifoAnimator::get_t() {
    return t;
}
