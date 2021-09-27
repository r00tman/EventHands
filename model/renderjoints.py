#!/usr/bin/env python3
import ffmpeg
import numpy as np
import sys
import os
import cv2 as cv
from tqdm import trange


WIDTH, HEIGHT = 240, 180


base = sys.argv[1][:-4]
jointsfn = base+'.txt'
camerafn = sys.argv[2]
overlayfn = base[:-len('_joints')]+'_overlay_rj.mp4'
jointsvideofn = base+'.mp4'
screenspacefn = base+'_ss.txt'
print(base, jointsfn, camerafn, overlayfn, jointsvideofn, screenspacefn, sep='\n')

joints = np.loadtxt(jointsfn)
joints = joints.reshape(joints.shape[0], -1, 3)
matrices = np.loadtxt(camerafn)
proj = matrices[0].reshape(4, 4)
view = matrices[1].reshape(4, 4)
# matrices = matrices[2:]
# matrices = matrices.reshape(matrices.shape[0], 4, 4)

print(proj)
print(view)

ones = np.ones((joints.shape[0], joints.shape[1], 1))
print(ones.shape, joints.shape)
joints = np.concatenate((joints, ones), 2)
print(joints.shape)
newjoints = np.zeros_like(joints)
for i in trange(joints.shape[0]):
    for j in range(joints.shape[1]):
        cj = joints[i, j]
        # cj = matrices[i] @ cj
        # print(cj[3])
        cj = view @ cj
        # print(cj[3])
        cj = proj @ cj
        # print(cj[3])
        newjoints[i, j] = cj/cj[3]
        # 1/0
newjoints = newjoints[:-1]
# newjoints = np.nan_to_num(newjoints)

joints = newjoints[:, :, :3]

# joints = (joints @ (proj @ view).T)[:, :, :3]
print(joints.shape)
joints = joints[14:]

which = [21, 52, 53, 54, 55, 56]
which = which + list(range(37, 52))
joints = joints[:, which]
# matrices = matrices[14:]

screenx = (joints[..., 0]+1)*WIDTH/2
screeny = (-joints[..., 1]+1)*HEIGHT/2
frameidx = np.arange(len(screenx))
screenpos = np.stack([screenx, screeny], 2)
print(screenpos.shape)
np.savetxt(screenspacefn, np.c_[frameidx, screenpos.reshape(len(screenpos), -1)])

# inp_pr = (
#     ffmpeg
#     # .input(sys.argv[1], hwaccel='cuda')
#     .input(overlayfn)
#     .output('pipe:', format='rawvideo', pix_fmt='rgb24')
#     .run_async(pipe_stdout=True)
# )

# FPS = 60

# out = (
#     ffmpeg
#     # .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(WIDTH, HEIGHT), framerate=FPS, hwaccel='cuda')
#     .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(WIDTH, HEIGHT), framerate=FPS)
#     # .output(sys.argv[1]+'_overlay.mp4', pix_fmt='yuv444p', crf=7, vcodec='h264_nvenc')
#     .output(jointsvideofn, pix_fmt='yuv444p', crf=7)
#     .overwrite_output()
#     .run_async(pipe_stdin=True)
# )

# cnt = 0
# while True:
#     f = np.zeros((HEIGHT, WIDTH, 3))
#     # if cnt % 2 == 0:
#     pro = inp_pr.stdout.read(WIDTH*HEIGHT*3)
#     if not pro:
#         break
#     pr = np.frombuffer(pro, np.uint8).reshape([HEIGHT, WIDTH, 3])
#     f = np.copy(pr)
#     # cj = joints[cnt, 20:22]
#     for i in range(joints.shape[1]):
#         # x, y, z = cj[i]
#         # x = (x + 1)/2
#         # y = (-y + 1)/2
#         x, y = screenpos[min(cnt, len(screenpos)-1), i]
#         f = cv.circle(f, (int(x), int(y)), 0, (255, 255, 255), 3)
#         f = cv.putText(f, str(i), (int(x), int(y)), 0, 0.20, (0, 255, 255))

#     out.stdin.write(
#         f
#         .astype(np.uint8)
#         .tobytes()
#     )
#     cnt += 1

# # np.savetxt(

# out.stdin.close()
# out.wait()
# inp_pr.stdout.close()
# inp_pr.wait()
