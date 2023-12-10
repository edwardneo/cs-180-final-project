"""
HDR stencil code - student.py
CS 1290 Computational Photography, Brown U.
"""

import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize


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
        l:      lambda, the constant that determines the amount of
                smoothness.
        w[z]:   the weighting function value for pixel value z (where z is between 0 - 255).

    Returns:
        g[z]:   the log exposure corresponding to pixel value z (where z is between 0 - 255).
        lE[i]:  the log film irradiance at pixel location i.
    """

    n = 256  
    p = Z.shape[0] * Z.shape[1]  

    A = np.zeros((p + n-1, n + Z.shape[0]))
    b = np.zeros((p + n-1,))

    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            wij = w[Z[i, j]]
            A[k, Z[i, j]] = wij
            A[k, n + i] = -wij
            b[k] = wij * B[i, j]
            k += 1

    A[k, n//2] = 1  
    k += 1

    for i in range(1, n-1):
        curr_w = w[i]
        A[k, i - 1] = l * curr_w
        A[k, i] = -2 * l * curr_w
        A[k, i +1] = l * curr_w
        k += 1

    x = np.linalg.lstsq(A, b, rcond=None)[0]
    
    g = x[:n]
    lE = x[n:]

    return g, lE



def hdr(file_names, g_red, g_green, g_blue, w, exposure_matrix, nr_exposures, to_align = False):
    """
    Given the imaging system's response function g (per channel), a weighting function
    for pixel intensity values, and an exposure matrix containing the log shutter
    speed for each image, reconstruct the HDR radiance map.

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

    # Initialize the HDR radiance map
    im_shape = cv2.imread(file_names[0]).shape
    gs = [g_red, g_green, g_blue]
    p = len(file_names)
    if not to_align:
        images = np.array([cv2.cvtColor(cv2.imread(file_names[k]), cv2.COLOR_BGR2RGB) / 255.0 for k in range(p)])
    else:
        temp = np.array([cv2.cvtColor(cv2.imread(file_names[k]), cv2.COLOR_BGR2RGB) / 255.0 for k in range(p)])
        images = temp.copy()
        src = images[0]
        for i in range(1, len(images)):
            pr = pyramid(src,temp[i])
            images[i] = align(temp[i], pr)

    r, c, _ = im_shape

    hdr = np.zeros((r, c, 3))

    for k in range(3):
        Z = images[:, :, :, k] * 255.0
        wZ = w[Z.astype(int)]
        valid_mask = np.sum(wZ, axis=0) > 0

        # Adjust the shapes to perform the weighted sum along the stack axis
        E = np.sum(wZ * (gs[k][Z.astype(int)] - exposure_matrix[0, :][:, None, None]), axis=0) / np.sum(wZ, axis=0)
        E[~valid_mask] = 0

        hdr[:, :, k] = np.exp(E)

    hdr /= nr_exposures

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

    return (hdr_radiance_map)/(1 + hdr_radiance_map)


def tm_durand(hdr_radiance_map, dR=4.0, d=5, sc=15, ss=15, gamma = 0.5):
    """
    Your implementation of:
    http://people.csail.mit.edu/fredo/PUBLI/Siggraph2002/DurandBilateral.pdf

    Args:
        hdr_radiance_map (np.array): HDR radiance map of the image
                                     with shape (H, W, 3)
    Returns:
        np.array of image with values in range [0.0, 1.0]
    """

    I =np.mean(hdr_radiance_map, axis = 2)

    chrominance = hdr_radiance_map /(I[..., None] + 1e-10)

    E = np.log2(I + 1e-10)

    B = cv2.bilateralFilter(E.astype(np.float32), d=d, sigmaColor=sc, sigmaSpace=ss)

    D = E - B

    offset = np.max(B)
    scale = dR / (np.max(B) - np.min(B))
    B_prime = (B - offset) * scale

    O = 2 ** (B_prime + D)

    result = O[..., None] * chrominance

    result = result ** gamma

    return result


# ========================================================================
# Bells and Whistles Align
# ========================================================================


def metric(reference, tofix):
    sizex, sizey, z = reference.shape
    cropx, cropy = int(sizex/10), int(sizey/10)
    
    return -(np.sum((reference[cropx:-cropx, cropy:-cropy,:]-tofix[cropx:-cropx, cropy:-cropy,:])**2))
    #
def find_position(reference, tofix, cx, cy, it):
    
    max_metric = metric(reference, tofix)
    position = (0,0)
    for i in range(cx -it, cx + it):
        temp = np.roll(tofix, i, axis = 0)
        for j in range(cy -it, cy + it):
            shifted = np.roll(temp, j, axis = 1)
            curr = metric(reference,shifted)
            if curr > max_metric:
                max_metric = curr
                position = (i,j)
    #print(max_metric)
    return position
def find_position_full(reference, tofix):
    max_metric = metric(reference, tofix)
    position = (0,0)
    for i in range(reference.shape[0]):
        temp = np.roll(tofix, i, axis = 0)
        for j in range(reference.shape[1]):
            shifted = np.roll(temp, j, axis = 1)
            curr = metric(reference,shifted)
            if curr > max_metric:
                max_metric = curr
                position = (i,j)
    #print(max_metric)
    return position
    

def align(image, position):
    temp = np.roll(image, position[0], axis = 0)
    return np.roll(temp, position[1], axis = 1)

def pyramid(reference, tofix):
    x,y, z = reference.shape
    it = 10
    if x > 20 or y > 20:
        res_r = resize(reference, (x/2, y/2, z))
        res_t = resize(tofix,(x/2, y/2, z))
        cx, cy = pyramid(res_r, res_t)
        cx *= 2
        cy *= 2
        return find_position(reference, tofix, cx, cy, it)
    else:
        return find_position_full(reference, tofix)