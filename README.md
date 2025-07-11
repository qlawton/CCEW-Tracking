# ccew_tracking

Python code to track wave objects and compute forecast statistics for tropical waves -- primarily Convectively Coupled Kelvin Waves (CCKWs), but with support for the "tropical depression (TD)" band and equatorial Rossby waves (n=1), in observational and model data.

These scripts were written in partial support for an in-progress manuscript:

"On the Representation of Kelvin Waves and Easterly Waves in Operational Forecast Models: An Object-Based Approach", by Quinton A. Lawton, Rosimar Rios-Berrios, Falko Judt, Linus Magnusson, and Martin Köhler.


## Folder Structure

- `example_data/`: Example NetCDF precipitation data for testing and demonstration.
- `example_shell_scripts/`: Example shell scripts for batch processing.
- `forecast_metrics/`: Scripts for quantifing forecast skill 
- `tracking_scripts/`: Main Python scripts for filtering, preprocessing, and tracking tropical waves.

## Main Scripts

- `tracking_scripts/preprocess_precipitation_for_filter.py`: Preprocesses model and observational precipitation data, slices to desired latitude/time, and pads for filtering.
- `tracking_scripts/FFT_filter_imerg_daily_mean.py`: Filters precipitation data in wavenumber-frequency space for specified wave types (Kelvin, TD, ER, etc.).
- `tracking_scripts/track_filtered_waves.py`: Tracks filtered wave systems (Kelvin, TD, ER) in precipitation data and outputs tracked longitude and strength.

## Supporting scripts
- `tracking_scripts/tropical_diagnostics/spacetime.py`: Adapted from Maria Gehne's "Tropical Diagnostics" github package (https://github.com/mgehne/tropical_diagnostics). This contains core functions for space-time spectral analysis and filtering.

## Usage

1. **Preprocess Data:** Use `preprocess_precipitation_for_filter.py` to prepare input data.
2. **Filter Data:** Apply `FFT_filter_imerg_daily_mean.py` to extract desired wave signals.
3. **Track Waves:** Run `track_filtered_waves.py` to identify and track wave systems.
4. **Forecast Diagnostics:** Use functions in `forecast_metrics` to compute forecast errors.

## Requirements

- Python 3.x
- xarray, numpy, pandas, scipy

For more details, see comments in each script.


