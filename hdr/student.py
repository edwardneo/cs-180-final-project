"""
HDR stencil code - student.py
CS 1290 Computational Photography, Brown U.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


# ========================================================================
# RADIANCE MAP RECONSTRUCTION
# ========================================================================


def solve_g(Z, B, l, w):
    """
    Given a set of pixel values observed for several pixels in several
    images with different exposure times, this function returns the
    imaging system's response function g as well as the log film irradiance
    values for the observed pixels.

    Args:
        Z[i,j]: the pixel values of pixel location number i in image j.
        B[i,j]: the log delta t, or log shutter speed, for image j at pixel i
                (will be the same value for each i within the same j).
        l       lamdba, the constant that determines the amount of
                smoothness.
        w[z]:   the weighting function value for pixel value z (where z is between 0 - 255).

    Returns:
        g[z]:   the log exposure corresponding to pixel value z (where z is between 0 - 255).
        lE[i]:  the log film irradiance at pixel location i.

    """

    import numpy as np
from scipy.optimize import minimize

def solve_g(Z, B, l, w):
    """
    Given a set of pixel values observed for several pixels in several
    images with different exposure times, this function returns the
    imaging system's response function g as well as the log film irradiance
    values for the observed pixels.

    Args:
        Z[i,j]: the pixel values of pixel location number i in image j.
        B[i,j]: the log delta t, or log shutter speed, for image j at pixel i
                (will be the same value for each i within the same j).
        l:      lambda, the constant that determines the amount of
                smoothness.
        w[z]:   the weighting function value for pixel value z (where z is between 0 - 255).

    Returns:
        g[z]:   the log exposure corresponding to pixel value z (where z is between 0 - 255).
        lE[i]:  the log film irradiance at pixel location i.
    """
    Z = np.clip(Z, 0, 255)  # Ensure pixel values are within the valid range

    n = 256  # Number of possible pixel values
    p = Z.shape[0] * Z.shape[1]  # Total number of observations

    # Initialize variables
    A = np.zeros((p + n + 1, n + p))
    b = np.zeros((p + n + 1,))
    w = np.vectorize(w)  # Vectorize the weighting function

    # Fill in A matrix and b vector
    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            wij = w(Z[i, j])
            A[k, Z[i, j]] = wij
            A[k, n + i] = -wij
            b[k] = wij * B[i, j]
            k += 1

    A[k, 127] = 1  # Fix the curve by setting the middle value to 0
    k += 1

    for i in range(1, n - 1):
        A[k, i - 1] = l * w(i)
        A[k, i] = -2 * l * w(i)
        A[k, i + 1] = l * w(i)
        k += 1

    # Solve the system of linear equations using least squares
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    g = x[:n]
    lE = x[n:]

    return g, lE


def hdr(file_names, g_red, g_green, g_blue, w, exposure_matrix, nr_exposures):
    """
    Given the imaging system's response function g (per channel), a weighting function
    for pixel intensity values, and an exposure matrix containing the log shutter
    speed for each image, reconstruct the HDR radiance map in accordance to section
    2.2 of Debevec and Malik 1997.

    Args:
        file_names:           exposure stack image filenames
        g_red:                response function g for the red channel.
        g_green:              response function g for the green channel.
        g_blue:               response function g for the blue channel.
        w[z]:                 the weighting function value for pixel value z
                              (where z is between 0 - 255).
        exposure_matrix[i,j]: the log delta t, or log shutter speed, for image j at pixel i
                              (will be the same value for each i within the same j).
        nr_exposures:         number of images / exposures

    Returns:
        hdr:                  the hdr radiance map.
    """

    n = 256  # Number of possible pixel values
    p = len(file_names)  # Number of images

    # Initialize variables
    hdr = np.zeros((len(g_red), len(g_red[0]), 3))  # Initialize HDR radiance map
    w = np.vectorize(w)  # Vectorize the weighting function

    for i in range(len(g_red)):
        for j in range(len(g_red[0])):
            Z = np.zeros((p,))  # Observed pixel values for the current pixel location

            # Collect pixel values for each exposure
            for k in range(p):
                img = cv2.imread(file_names[k])
                Z[k] = img[i, j, 0]  # Assuming the image is in BGR format

            # Calculate the radiance using the inverse response function
            E_red = np.sum(w(Z) * (g_red[Z.astype(int)] - exposure_matrix[i, :])) / np.sum(w(Z))
            E_green = np.sum(w(Z) * (g_green[Z.astype(int)] - exposure_matrix[i, :])) / np.sum(w(Z))
            E_blue = np.sum(w(Z) * (g_blue[Z.astype(int)] - exposure_matrix[i, :])) / np.sum(w(Z))

            hdr[i, j, 0] = np.exp(E_red)
            hdr[i, j, 1] = np.exp(E_green)
            hdr[i, j, 2] = np.exp(E_blue)

    return hdr


# ========================================================================
# TONE MAPPING
# ========================================================================


def tm_global_simple(hdr_radiance_map):
    """
    Simple global tone mapping function (Reinhard et al.)

    Equation:
        E_display = E_world / (1 + E_world)

    Args:
        hdr_radiance_map (np.array): HDR radiance map of the image
                                     with shape (H, W, 3)
    Returns:
        np.array of image with values in range [0.0, 1.0]
    """

    # Extract the luminance channel (assuming RGB format)
    luminance = 0.2126 * hdr_radiance_map[:, :, 0] + 0.7152 * hdr_radiance_map[:, :, 1] + 0.0722 * hdr_radiance_map[:, :, 2]

    # Apply tone mapping operator
    tone_mapped = luminance / (1 + luminance)

    # Extend to three channels for the final tone-mapped image
    tone_mapped_image = np.stack([tone_mapped] * 3, axis=-1)

    return tone_mapped_image


def tm_durand(hdr_radiance_map):
    """
    Your implementation of:
    http://people.csail.mit.edu/fredo/PUBLI/Siggraph2002/DurandBilateral.pdf

    Args:
        hdr_radiance_map (np.array): HDR radiance map of the image
                                     with shape (H, W, 3)
    Returns:
        np.array of image with values in range [0.0, 1.0]
    """

    # Extract luminance
    luminance = 0.2126 * hdr_radiance_map[:, :, 0] + 0.7152 * hdr_radiance_map[:, :, 1] + 0.0722 * hdr_radiance_map[:, :, 2]

    # Compute base layer using bilateral filter
    base_layer = cv2.bilateralFilter(luminance.astype(np.float32), d=0, sigmaColor=range_sigma, sigmaSpace=spatial_sigma)

    # Compute detail layer
    detail_layer = luminance - base_layer

    # Adjust base layer contrast
    base_layer = (base_layer - np.min(base_layer)) / (np.max(base_layer) - np.min(base_layer))
    base_layer = np.power(base_layer, base_contrast)

    # Combine base and detail layers
    tone_mapped = base_layer + detail_factor * detail_layer

    # Extend to three channels for the final tone-mapped image
    tone_mapped_image = np.stack([tone_mapped] * 3, axis=-1)

    return tone_mapped_image
