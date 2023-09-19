# Demosaicing RGB image with Python using Bilinear Interpolation

This toolkit provides utilities to process raw images, particularly those that have a Bayer pattern. The functions in this toolkit allow you to read raw TIFF images, display them, extract channels, normalize intensities, perform white balancing, and demosaic the image using bilinear interpolation.

## Features

1. **TIFF Image Reader**: Pulls a raw TIFF image and displays it.
2. **Bayer Decoder**: Extracts Red, Green, and Blue channels from an image that uses the RGGB Bayer pattern.
3. **White Balancing**: Applies white balancing to the image using user-defined RGB scales.
4. **Normalization**: Provides normalization functionalities for images.
5. **Bilinear Demosaicing**: Performs bilinear interpolation demosaicing on a raw image.
6. **Channel Visualization**: Displays the individual R, G, and B channels and combined Bayer pattern.

## Dependencies

- matplotlib
- cv2 (OpenCV for Python)
- numpy
- scipy
- Pillow (PIL)

## Installation

Before you can run the toolkit, you'll need to install the required dependencies. You can do this using either `conda` (recommended for managing Python environments) or `pip`.

## Using conda

If you don't have `conda` installed, you can get it through [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Once `conda` is set up, you can create an environment and install the dependencies as follows:


### Create a new environment named 'demosaic'
```
conda create --name demosaic python=3.8
```
### Activate the environment
```
conda activate demosaic
```
### Install dependencies
```
conda install -c conda-forge matplotlib
```
```
conda install -c anaconda numpy
```
```
conda install -c anaconda scipy
```
```
conda install -c anaconda pillow
```
```
conda install -c conda-forge opencv
```

## Using PIP
If you prefer using pip, you can install the dependencies directly:
'''
pip install matplotlib numpy scipy Pillow opencv-python
'''

## Usage

1. Make sure the required dependencies are installed.
2. Place your raw TIFF image in the same directory as the script and name it 'raw_image.tiff'.
3. Run the script. This will read the image, display various visualizations (raw image, individual channels, Bayer pattern, demosaiced image), and perform white balancing and bilinear demosaicing.

## Functions Overview

- `pull_image()`: Reads and displays a TIFF image named 'raw_image.tiff' and returns its data.
- `normalize_uint8(img, maxval, minval)`: Normalizes a uint16 image to a uint8 scale.
- `min_max_normalization(img, maxval, minval)`: General normalization function.
- `whitebalance(im, rgbScales)`: White balances an image using provided RGB scales.
- `bayer(im)`: Decodes an RGGB Bayer pattern image.
- `bilinear(im)`: Applies bilinear interpolation demosaicing on an image.
- Visualization functions like `display_raw_image(raw_image_path)`, `display_bayer_pattern(im)`, `display_red_channel(image)`, etc. are used to visualize various stages of the raw image processing.

### Notes

The toolkit assumes that the input image is in RGGB Bayer pattern. Adjustments may be needed for other patterns.


### Contact

[https://github.com/akhawaja2014]


