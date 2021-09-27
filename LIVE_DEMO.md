# How to run live demo

1. Download model.ckpt from https://gvv-assets.mpi-inf.mpg.de/EventHands/ to `live_demo/`.
2. In DV, clear the graph, add the event camera node, add a network node. Connect events in the camera output to the network node input.

Note:
You can save the DV graph as `.xml` and load it later instead of recreating it every time.

Note:
You can use a prerecorded `.aedat4` file instead of an event camera. Instead of an event camera node, use the aedat4 reader node.

3. Compile the native module for the live prediction code. More info is in the corresponding README.md.
4. Run `live_predict.py`. This will capture DV event stream via network and output the predictions to `/tmp/preds` FIFO.
5. Run the event simulator compiled from `evsim_live` using the predicted outputs at `/tmp/preds` (run at `evsim_joints` working directory where all the simulator assets are):
```
../evsim_live_build/evsim -p fifo:///tmp/preds
```

You should be able to see the live predictions on the screen.

