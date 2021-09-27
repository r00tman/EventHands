# High-Performance Event Camera Simulator for MANO/SMPL

## Hacking
This is a regular CMake project, so it should work normally in that regard.

Since Nvidia didn't support gcc-9, please use gcc-8, e.g., set these CMake options in case your default compiler is gcc-9:
```
-DCMAKE_CXX_COMPILER=<path-to-g++-8> -DCMAKE_C_COMPILER=<path-to-gcc-8>
```

Please download and unpack all the assets from https://gvv-assets.mpi-inf.mpg.de/EventHands/

Also, please download the texturing basis files, i.e., ```mean_data_vec.bin```, ```eigen_vector_matrix.bin```, and ```std_dev_matrix.bin``` to the project directory from
https://drive.google.com/open?id=1F30omQzGS4KyOc1ekDnyoOMw52g5l8dK

My IDE of choice for this project is Qt Creator for its stellar fakevim mode, CMake, CUDA, GLSL support with built-in CLang code completion. But one is free to use any other option they like; the project is CMake-based without any specific IDE files.

The project is developed on Linux. Windows compatibility is not tested, but there is no platform-specific code in there, so it might work out of the box.

## Dependencies
- [glew](http://glew.sourceforge.net/)
- [glm](https://glm.g-truc.net/)
- [SDL2](https://www.libsdl.org)
- [xtensor](https://github.com/xtensor-stack/xtensor), [xtensor-blas](https://github.com/xtensor-stack/xtensor-blas), [xtensor-io](https://github.com/xtensor-stack/xtensor-io)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

## Usage manual
Please run the executable not from the build directory, but rather from the source directory since it looks for shaders and models there.

If you're using the project on a Nvidia Optimus laptop, you need to execute it, e.g., via ```optirun``` or ```primusrun``` so both OpenGL and CUDA would be running on the Nvidia card.

To look for the options you can run `evsim --help`

Here are several keybinds you can use:
- ```q``` ```ESC``` - quit,
- ```v``` - toggle between generated event view and rendered scene view (default is event view),
- ```g``` - toggle between gamma-corrected and linear color in the rendered scene view (default is gamma-corrected),
- ```r``` - toggle off-screen rendering (default is off),
- ```a``` - toggle animation on/off (default is on),
- ```b``` - change betas (hand shape),
- ```c``` - toggle between CUDA and CPU MANO implementations (default is CPU),
- ```9``` ```0``` - decrease/increase the event threshold by 1.5 (default is 0.2),
- ```7``` ```8``` - decrease/increase simulated fps (default is 1000 fps),
- ```6``` - set simulated fps to 1000 (like in event camera),
- ```5``` - set simulated fps to 60 (real-time, e.g., for checking the animation).
