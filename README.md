# Demosaicing RGB image with Python using Bilinear Interpolation

Demosaicing is a digital image process used to reconstruct a full-color image from the incomplete color samples output by an image sensor overlaid with a color filter array (CFA). It is an essential step in the processing of raw images in any digital camera.

This code provides utilities to process raw images. The functions in this code allow you to read raw TIFF images, display them, extract channels, normalize intensities, perform white balancing, and **demosaic** the image using bilinear interpolation.

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
  Here is the raw mosaiced image. It is how the camera sees the World. 
  
  ![Alt text](https://github.com/akhawaja2014/Hands-on-Demosaicing-experiment-with-Python/blob/master/figures/RawImage.png)
  
  If you zoom enough in on the image, You will find the mosaiced pattern. Each square is one of the three channels' pixel values measured.
  Here is the zoomed image:

  ![Alt text](https://github.com/akhawaja2014/Hands-on-Demosaicing-experiment-with-Python/blob/master/figures/raw_image_zoomed.png)

  
- `normalize_uint8(img, maxval, minval)`: Normalizes a uint16 image to a uint8 scale.
  The Raw image is captured by the camera in 16-bit format.  In a 16-bit image, each pixel can represent $2^{16}  = 65,536 $ different colors or shades. This provides a much larger space than 8-bit images (256 colors) and allows for more subtle variations in color and   tone.
- `min_max_normalization(img, maxval, minval)`: General normalization function.
- `whitebalance(im, rgbScales)`: White balances an image using provided RGB scales.
- `bayer(im)`: Decodes an RGGB Bayer pattern image.
   The Bayer pattern, also known as the Bayer filter, is a color filter array (CFA) used in many digital imaging devices, including digital cameras and smartphone cameras. It's named after its inventor, Bryce Bayer, who developed this pattern while working at Eastman     Kodak in the 1970s. The Bayer pattern is a common method for capturing and reproducing color in digital images. The Bayer pattern consists of a grid of color filters placed over the image sensor's pixels. Each pixel in the sensor is covered by one of these color       filters, which are typically red, green, or blue. The Bayer pattern usually consists of 50% green filters, 25% red filters, and 25% blue filters, arranged in a specific repeating pattern. The most common arrangement for a Bayer filter is as follows, where "R"          stands for red, "G" stands for green, and "B" stands for blue. Here is what the Bayer Color Filter Array (CFA) looks like:

    ![Bayer Pattern](https://github.com/akhawaja2014/Hands-on-Demosaicing-experiment-with-Python/blob/master/figures/bayercfa.PNG)

    The Bayer Pattern for the raw image looks like this. The intensities of the color represent the amount of light that has been captured by the sensor in that respective channel.
  
    ![Bayer Pattern](https://github.com/akhawaja2014/Hands-on-Demosaicing-experiment-with-Python/blob/master/figures/Bayer_pattern.png)

    If you zoom this image, you can clearly see the Bayer Pattern.

    ![Bayer Pattern zoomed](https://github.com/akhawaja2014/Hands-on-Demosaicing-experiment-with-Python/blob/master/figures/Bayer_pattern_zoomed.png)

- Visualization functions like `display_raw_image(raw_image_path)`, `display_bayer_pattern(im)`, `display_red_channel(image)`, etc. are used to visualize various stages of the raw image processing.
    Here we now plot each channel of bayer pattern seperately for understanding. Here is how Green Channel looks like. 

    ![Green Channel](https://github.com/akhawaja2014/Hands-on-Demosaicing-experiment-with-Python/blob/master/figures/green_channel.png)

    Here is the zoomed version of Green Channel.

    ![Green Channel zoomed](https://github.com/akhawaja2014/Hands-on-Demosaicing-experiment-with-Python/blob/master/figures/green_channelzoomed.png)

    Here is the Red Channel.

    ![Red Channel](https://github.com/akhawaja2014/Hands-on-Demosaicing-experiment-with-Python/blob/master/figures/red_channel.png)

    Here is the zoomed version of Red Channel.

    ![Red Channel Zoomed](https://github.com/akhawaja2014/Hands-on-Demosaicing-experiment-with-Python/blob/master/figures/red_channel_zoomed.png)

    Here is the Blue Channel.

    ![Blue Channel](https://github.com/akhawaja2014/Hands-on-Demosaicing-experiment-with-Python/blob/master/figures/blue_channel.png)

    Here is the zoomed version of Blue Channel.

    ![Blue Channel Zoomed](https://github.com/akhawaja2014/Hands-on-Demosaicing-experiment-with-Python/blob/master/figures/blue_channel_zoomed.png)


   
- `bilinear(im)`: Applies bilinear interpolation demosaicing on an image.

    Here is the image after Bilinear demosaicing.

   ![demosaiced](https://github.com/akhawaja2014/Hands-on-Demosaicing-experiment-with-Python/blob/master/figures/demosaiked_image.png)

   Here is the zoomed version of demosaiced image.

  ![demosaiced zoom](https://github.com/akhawaja2014/Hands-on-Demosaicing-experiment-with-Python/blob/master/figures/demosaiked_zoom.png)
  




### Contact

[https://github.com/akhawaja2014]


