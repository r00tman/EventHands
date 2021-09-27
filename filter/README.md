# Kalman filtering scripts

## Dependencies
 - NumPy
 - [FilterPy](https://github.com/rlabbe/filterpy)

## Usage
```
% ./filter.py predictions.txt
% ./filter_fast.py predictions.txt
```

This will result in `predictions_filtered.txt` and `predictions_filteredfast.txt` files.

The fast filter is used for the slow motion sections of the supplementary video.
The regular filter is used for all other non-live-demo sections of the supplementary video.
