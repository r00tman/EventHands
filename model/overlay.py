#!/usr/bin/env python3
import ffmpeg
import numpy as np
import sys


WIDTH, HEIGHT = 240, 180

inp_pr = (
    ffmpeg
    # .input(sys.argv[1], hwaccel='cuda')
    .input(sys.argv[1])
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run_async(pipe_stdout=True)
)

inp_ev = (
    ffmpeg
    # .input(sys.argv[2], hwaccel='cuda')
    .input(sys.argv[2])
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run_async(pipe_stdout=True)
)

FPS = 60

out = (
    ffmpeg
    # .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(WIDTH, HEIGHT), framerate=FPS, hwaccel='cuda')
    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(WIDTH, HEIGHT), framerate=FPS)
    # .output(sys.argv[1]+'_overlay.mp4', pix_fmt='yuv444p', crf=7, vcodec='h264_nvenc')
    .output(sys.argv[1][:-4]+'_overlay.mp4', pix_fmt='yuv444p', crf=7)
    .overwrite_output()
    .run_async(pipe_stdin=True)
)
# for _ in range(14):
#     __ = inp_pr.stdout.read(WIDTH*HEIGHT*3)

cnt = 0
while True:
    f = np.zeros((HEIGHT, WIDTH, 3))
    # if cnt % 2 == 0:
    pro = inp_pr.stdout.read(WIDTH*HEIGHT*3)
    ev = inp_ev.stdout.read(WIDTH*HEIGHT*3)
    if not pro or not ev:
        break
    pr = np.frombuffer(pro, np.uint8).reshape([HEIGHT, WIDTH, 3])
    ev = np.frombuffer(ev, np.uint8).reshape([HEIGHT, WIDTH, 3])
    f = np.copy(pr)
    alpha = np.max(ev,2,keepdims=True)/255*0.5
    f = f*(1-alpha)+ev*alpha
    # mask = (alpha > 0)
    # f[mask] = ev[mask]
    out.stdin.write(
        f
        .astype(np.uint8)
        .tobytes()
    )
    cnt += 1

out.stdin.close()
out.wait()
inp_pr.stdout.close()
inp_ev.stdout.close()
inp_pr.wait()
inp_ev.wait()
