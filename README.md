# EventHands: Real-Time Neural 3D Hand Pose Estimation from an Event Stream

[Project Page](https://4dqv.mpi-inf.mpg.de/EventHands)

# Index
 - [TRAIN.md](https://github.com/r00tman/EventHands/blob/main/TRAIN.md) -- how to train the model from scratch
 - [EVAL_REAL.md](https://github.com/r00tman/EventHands/blob/main/EVAL_REAL.md) -- how to evaluate the model on the real data
 - [EVAL_SYNTH.md](https://github.com/r00tman/EventHands/blob/main/EVAL_SYNTH.md) -- how to evaluate the model on the synthetic data
 - [LIVE_DEMO.md](https://github.com/r00tman/EventHands/blob/main/LIVE_DEMO.md) -- how to run the live demo

# Data and Assets
Please download all the data from [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/DBJHfoMApyK53S2) into the corresponding folders.

If you don't plan retraining the network, you might want to skip downloading `train_data` as it is magnitutes larger than all of the other folders.

If you don't plan to use the evaluation datasets, you might also want to skip downloading `real_eval_data` and `synth_eval_data` folders.

# Environment
Please find the Anaconda environment we used for everything (incl. compiling the simulator) at `environment.yml`

# Models
 - `regular.ckpt` -- regular model
 - `bottom.ckpt` -- model for the hand coming from the bottom edge of the frame

# License
Permission is hereby granted, free of charge, to any person or company obtaining a copy of this software and associated documentation files (the "Software") from the copyright holders to use the Software for any non-commercial purpose. Publication, redistribution and (re)selling of the software, of modifications, extensions, and derivates of it, and of other software containing portions of the licensed Software, are not permitted. The Copyright holder is permitted to publically disclose and advertise the use of the software by any licensee.

Packaging or distributing parts or whole of the provided software (including code, models and data) as is or as part of other software is prohibited. Commercial use of parts or whole of the provided software (including code, models and data) is strictly prohibited. Using the provided software for promotion of a commercial entity or product, or in any other manner which directly or indirectly results in commercial gains is strictly prohibited.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
