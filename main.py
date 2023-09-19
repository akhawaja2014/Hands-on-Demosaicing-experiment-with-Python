from matplotlib import pyplot as plt
import cv2
import numpy as np
from scipy.signal import convolve2d
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap


def pull_image():
    """
    Reads a TIFF image named 'sample.tiff', displays it as a grayscale image, and returns its data as a numpy array.
    
    Returns:
    - numpy.ndarray: The image data as a 2D numpy array.
    """

    # Open the image 'sample.tiff' using the Image module.
    raw_data = Image.open('raw_image.tiff')

    # Convert the image data into a numpy array and cast the datatype to double (float64).
    raw = np.array(raw_data).astype(np.double)
    
    # Display the image using matplotlib with a grayscale colormap.
    plt.imshow(raw, cmap='gray')  # Assuming it's a grayscale image
    plt.title('Raw Image')
    plt.axis('off')
    plt.show()

    # Return the image data as a 2D numpy array.
    return raw


# Normalization
def normalize_uint8(img, maxval, minval):
    """
    img: uint16 2d raw image
    out: uint8 2d normalized 0-255 image
    """
    return (np.rint((img - img.min()) * ((maxval - minval) / (img.max() - img.min())) + minval)).astype(dtype='uint8')

def min_max_normalization(img, maxval, minval):
    """
    To normalize the the values
    """
    return (img - minval) / (maxval-minval)


# White balancing
def whitebalance(im, rgbScales):

    # generate the white balancing matrix
    scalematrix = rgbScales[1] * np.ones(im.shape)
    
    # rggb
    scalematrix[0::2, 0::2] = rgbScales[0]
    scalematrix[1::2, 1::2] = rgbScales[2]
    
    return np.multiply(im, scalematrix)
    

# Color filtering: `rggb`
def bayer(im):
    """
    
    Decodes an image using the RGGB Bayer pattern to extract and separate the red, green, and blue channels.
    
    """

    # Initialize a red channel matrix filled with zeros, having the same dimensions as the input image.
    r = np.zeros(im.shape[:2])

    # Initialize a green channel matrix filled with zeros, having the same dimensions as the input image.
    g = np.zeros(im.shape[:2])

    # Initialize a blue channel matrix filled with zeros, having the same dimensions as the input image.
    b = np.zeros(im.shape[:2])

    # For every alternate pixel in rows and columns, extract red channel values from the original image to the initialized red matrix.
    r[0::2, 0::2] += im[0::2, 0::2]

    # For every 'alternate' pixel in rows but adjacent pixel in columns, extract green channel values from the original image to the initialized green matrix.
    g[0::2, 1::2] += im[0::2, 1::2]

    # For every 'adjacent' pixel in rows but alternate pixel in columns, extract green channel values from the original image to the initialized green matrix.
    g[1::2, 0::2] += im[1::2, 0::2]

    # For every adjacent pixel in both rows and columns, extract blue channel values from the original image to the initialized blue matrix.
    b[1::2, 1::2] += im[1::2, 1::2]

    # Return the separated red, green, and blue channels.
    return r, g, b


# Demosaicing
def bilinear(im):

    """
    Interpolate the channels of an image using bilinear interpolation based on a given Bayer pattern.

    Parameters:
    - im (numpy.ndarray): The image to be interpolated.

    Returns:
    - tuple: Interpolated red, green, and blue channels as 2D matrices.
    """
    # GREEN FIRST
    # Decode the Bayer pattern of the image to get the red, green, and blue channels.
    r, g, b = bayer(im)

    # Define a kernel for green interpolation
    k_g = 1/4 * np.array([[0,1,0],[1,0,1],[0,1,0]])
    # Convolve the green channel with the kernel to get the interpolated values.
    convg =convolve2d(g, k_g, 'same')
    # Update the green channel with the interpolated values.
    g = g + convg

    # Define a kernel for initial red interpolation.
    k_r_1 = 1/4 * np.array([[1,0,1],[0,0,0],[1,0,1]])
    # Convolve the red channel with the initial kernel.
    convr1 =convolve2d(r, k_r_1, 'same')
    # Perform a secondary convolution using the green kernel on the updated red channel.
    convr2 =convolve2d(r+convr1, k_g, 'same')
    # Update the red channel with both sets of interpolated values.
    r = r + convr1 + convr2

    # Define a kernel for initial blue interpolation (same as the red kernel).
    k_b_1 = 1/4 * np.array([[1,0,1],[0,0,0],[1,0,1]])
    # Convolve the blue channel with the initial kernel.
    convb1 =convolve2d(b, k_b_1, 'same')
    # Perform a secondary convolution using the green kernel on the updated blue channel.
    convb2 =convolve2d(b+convb1, k_g, 'same')
    # Update the blue channel with both sets of interpolated values.
    b = b + convb1 + convb2
    # Return the interpolated red, green, and blue channels.
    return r, g, b




def display_raw_image(raw_image_path):
    try:
        # Read the raw image using OpenCV
        raw_image = cv2.imread(raw_image_path, cv2.IMREAD_UNCHANGED)

        if raw_image is not None:
            # Display the raw image
            plt.imshow(raw_image, cmap='gray')  # Assuming it's a grayscale image
            plt.title('Raw Image')
            plt.axis('off')
            plt.show()
        else:
            print("Failed to read the raw image.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def display_bayer_pattern(im):
    r, g, b = bayer(im)
    
    # Stacking the channels for visualization
    stacked_image = np.stack((r,g,b), axis=2).astype(np.uint8)
    
    # Displaying the Bayer pattern image
    plt.imshow(stacked_image)
    plt.title("Bayer Pattern")
    plt.axis('off')
    plt.show()

def display_red_channel(image):
    r, _, _ = bayer(image)  # Extract red channel

    # Create a custom red colormap
    cdict = {'red':   [(0.0,  0.0, 0.0),
                       (1.0,  1.0, 1.0)],
             'green': [(0.0,  0.0, 0.0),
                       (1.0,  0.0, 0.0)],
             'blue':  [(0.0,  0.0, 0.0),
                       (1.0,  0.0, 0.0)]}
    red_cmap = LinearSegmentedColormap('Reds', cdict)
    
    # Display the red channel using the custom colormap
    plt.imshow(r, cmap=red_cmap)
    plt.title("Red Channel")
    plt.colorbar()
    plt.show()

def display_green_channel(image):
    _, g, _ = bayer(image)
    green_cmap = LinearSegmentedColormap.from_list(
        'green_cmap', [(0, 0, 0), (0, 1, 0)], N=256
    )
    print(g)
    plt.imshow(g, cmap=green_cmap)
    plt.title('Green Channel')
    plt.axis('off')
    plt.colorbar()
    plt.show()

def display_blue_channel(image):
    _, _, b = bayer(image)
    blue_cmap = LinearSegmentedColormap.from_list(
        'blue_cmap', [(0, 0, 0), (0, 0, 1)], N=256
    )
    print(b)
    plt.imshow(b, cmap=blue_cmap)
    plt.title('Blue Channel')
    plt.axis('off')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":


    black = 0
    white = 65535

    
    R_scale = 1.000000
    G_scale = 1.000000
    B_scale = 1.000000

    im = pull_image()

    # Display the Bayer patterns
    display_red_channel(im)
    display_green_channel(im)
    display_blue_channel(im)

    display_bayer_pattern(im)

    im_norm = min_max_normalization(im, white, black)

    im_wb = whitebalance(im_norm, rgbScales = [R_scale, G_scale, B_scale])

    r, g, b = bilinear(im_wb)

    image = np.stack((r,g,b), axis=2)
    plt.axis('off')
    plt.title("Bilinear interpolation demosaicing")
    plt.imshow(image)
    plt.show()

