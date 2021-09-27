# Tool for Syncing aedat4 Events (visualized as LNES) With Videos

## Installation
Just install the dependencies.

## Dependencies
- numpy
- PyQt5
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python)
- dv
- opencv
- tqdm
- scipy
- matplotlib

## Usage
```
% ./main.py events.aedat4 video.mp4 <ACTUAL_VIDEO_FPS>
```
`ACTUAL_VIDEO_FPS` is, e.g., 500, if your camera shoots at 500 FPS.

Adjust the event offset to match the video on the left, then use `Save Trimmed Events...` to save the resulting `.npz` file that contains the events trimmed to video, syncing offsets, and the source files.
