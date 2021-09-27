# High-Performance evsim Output Reader

## Installation
```
% python setup.py develop
```

## Dependencies
- Python
- NumPy

## Usage
```
import evcreader

FILE_NAME = 'example.evc'
N_FRAMES = 1000
result = evcreader.read(FILE_NAME, N_FRAMES)
```

This will load the first `N_FRAMES` of frames from `FILE_NAME`. `result` will be an array of NumPy arrays, one for each frame loaded. These NumPy arrays will contain all events in the corresponding frame. These arrays will consist of triples `(x, y, polarity)` using `uint8_t` data type.

