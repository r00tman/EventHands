# How to evaluate the model on the synthetic data

1. Run prediction code:
```
% ./predict3_synth.py datasetname predictions.txt
```

2. Filter the output:
```
% ./filter.py predictions.txt
% ./filter_fast.py predictions.txt
```

This will result in `predictions_filtered.txt` and `predictions_filteredfast.txt` files.

The fast filter is used for the slow motion sections of the supplementary video.
The regular filter is used for all other non-live-demo sections of the supplementary video.

3. Extract the joint locations, e.g., from the fast filtered predictions, by running the event simulator compiled from `evsim_joints` (run at `evsim_joints` working directory, where all the simulator assets are):
```
% ../evsim_joints_build/evsim -o -p predictions_filteredfast.txt -r 1000 -j joints.txt -w 240 -h 180 -v | ffmpeg -y -s 240x180 -f rawvideo -pix_fmt rgb24 -framerate 60 -i pipe: -vf vflip -crf 1 -pix_fmt yuv444p video1000.mp4
```
This will produce `joints.txt` and `joints_camera.txt` files containing 3D joint locations and the used virtual camera parameters. Also this will visualize the predictions into `video1000.mp4`. Every single prediction corresponds to one frame. Thus it plays at `1000 FPS / 60 FPS = 16.667` times slower than the real-time. 

If one wants to visualize something at 60 FPS, one can either set the `-r 60` instead of `-r 1000` or speed the video up by `1000/60` times, e.g., using ffmpeg. With the first method, the joints file will only contain values corresponding to each 60 FPS frame instead of the full 1000 FPS frames.

4. Convert the joints to the screen space:
```
% ./render_joints.py joints.txt joints_camera.txt
```

This will produce `joints_ss.txt` file with joint positions in screen-space.

5. Compute the errors and plot the results:
```
% ./eval_synth_model_2d.py joints_ss.txt synthjoints/synthdemo_joints_ss.txt result_2d.pdf
% ./eval_synth_model_3d.py joints.txt synthjoints/synthjoints_joints.txt result_3d.pdf
```
