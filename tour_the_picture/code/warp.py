import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import skimage.io as skio
from scipy.interpolate import RegularGridInterpolator

import cv2 as cv

def homog(pts):
    pts = np.array(pts)
    assert pts.shape[1] == 2, f'Points are of shape {pts.shape}'
    
    return np.vstack((pts.T, np.ones((1, pts.shape[0]))))

def inv_homog(pts):
    pts = np.array(pts)
    assert pts.shape[0] == 3, f'Points are of shape {pts.shape}'
    assert np.all(pts[2, :] == 1), f'Points are not homogenous with final row {pts[2, :]}'

    return pts[:2, :].T

def normalize_homog(pts):
    pts = np.array(pts)
    assert pts.shape[0] == 3, f'Points are of shape {pts.shape}'

    return pts / pts[2, :]

def get_corners(im):
    height, width, _ = im.shape
    return np.array([(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)])

def get_range(corners):
    return (int(np.min(corners[:, 1])), int(np.max(corners[:, 1])) + 1), (int(np.min(corners[:, 0])), int(np.max(corners[:, 0])) + 1)

def get_translation(range):
    return np.array((range[1][0], range[0][0]))

def get_bounding_box(range):
    return np.zeros((range[0][1] - range[0][0], range[1][1] - range[1][0], 3))

def computeH(corr1, corr2):
    corr1, corr2 = np.array(corr1), np.array(corr2)
    
    assert corr1.shape == corr2.shape, f"Points do not match with shapes {corr1.shape} and {corr2.shape}"
    assert corr1.shape[1] == corr2.shape[1] == 2, f"Points are not 2D with shapes {corr1.shape} and {corr2.shape}"
    assert corr1.ndim == corr2.ndim == 2, f"Point arrays have too many dimensions with shapes {corr1.shape} and {corr2.shape}"
    
    N = corr1.shape[0]

    A_even_rows = np.hstack((corr1, np.ones((N, 1)), np.zeros((N, 3)), -corr1 * corr2[:, 0][:, np.newaxis]))
    A_odd_rows = np.hstack((np.zeros((N, 3)), corr1, np.ones((N, 1)), -corr1 * corr2[:, 1][:, np.newaxis]))
    
    A = np.hstack((A_even_rows, A_odd_rows)).reshape((-1, 8))
    b = corr2.flatten()
    
    H_vec = np.concatenate((np.linalg.lstsq(A, b, rcond=-1)[0], np.ones(1)))
    H = H_vec.reshape(3, 3)
    return H

# def warp_image(im, H):
#     corners = get_corners(im)
#     new_corners = inv_homog(normalize_homog(H @ homog(corners)))
#     print(new_corners)

#     range = get_range(new_corners)
    
#     translation = get_translation(range)
#     new_im = get_bounding_box(range)

#     new_pixels = np.vstack(sk.draw.polygon(new_corners[:, 0] - translation[0], new_corners[:, 1] - translation[1])).T
#     new_pixels_before_trans = np.vstack((new_pixels[:, 0] + translation[0], new_pixels[:, 1] + translation[1])).T
#     old_pixels = inv_homog(normalize_homog(np.linalg.inv(H) @ homog(new_pixels_before_trans)))
    
#     im_interp = RegularGridInterpolator((np.arange(im.shape[1]), np.arange(im.shape[0])), np.transpose(im, (1, 0, 2)), bounds_error=False, fill_value=0.5)
#     pixel_vals = im_interp(old_pixels)

#     new_im[new_pixels[:, 1], new_pixels[:, 0]] = pixel_vals
    
#     return new_im, new_corners, translation, new_corners

# def warp_image(im, H):
#     height, width, channels = im.shape
#     corners = get_corners(im)
#     new_corners = inv_homog(normalize_homog(H @ homog(corners)))

#     range = get_range(new_corners)
    
#     translation = get_translation(range)
#     new_im = get_bounding_box(range)
#     new_height, new_width, _ = new_im.shape

#     new_pixels = np.vstack(sk.draw.polygon(new_corners[:, 0] - translation[0], new_corners[:, 1] - translation[1])).T
#     new_pixels_before_trans = np.vstack((new_pixels[:, 0] + translation[0], new_pixels[:, 1] + translation[1])).T
#     old_pixels = inv_homog(normalize_homog(np.linalg.inv(H) @ homog(new_pixels_before_trans)))
    
#     im_interp = RegularGridInterpolator((np.arange(im.shape[1]), np.arange(im.shape[0])), np.transpose(im, (1, 0, 2)), bounds_error=False, fill_value=1)
#     pixel_vals = im_interp(old_pixels)

#     new_im[new_pixels[:, 1], new_pixels[:, 0]] = pixel_vals
    
#     return new_im, new_corners, translation

# def rectify(im, corr1, corr2):
#     H = computeH(corr2, corr1)
#     print(H)
#     new_im, _, translation, new_corners = warp_image(im, H)
#     print(new_im.shape)
#     return new_im, translation, new_corners
    # return new_im[translation[0] : translation[0] + int(corr2[2][1]), translation[1] : translation[1] + int(corr2[2][0])], translation, new_pixels

def rectify(im, corr_im, rect_dim):
    im_interp = RegularGridInterpolator((np.arange(im.shape[1]), np.arange(im.shape[0])), np.transpose(im, (1, 0, 2)), bounds_error=False, fill_value=0.5)

    new_im = np.zeros((rect_dim[1], rect_dim[0], 3))
    rect_vertices = np.array(get_corners(new_im))
    pixel_coords = np.array(sk.draw.polygon(rect_vertices[:, 0], rect_vertices[:, 1])).T
    print(pixel_coords)
    print(rect_vertices)
    H = computeH(rect_vertices, corr_im)
    # assert np.allclose(H, cv.findHomography(rect_vertices, corr_im)[0])
    print('rect', rect_vertices)
    print('corr_im', corr_im)
    print('f', inv_homog(normalize_homog(H @ homog(rect_vertices))))
    reference_coords = inv_homog(normalize_homog(H @ homog(pixel_coords)))
    pixel_vals = im_interp(reference_coords)

    new_im[pixel_coords[:, 1], pixel_coords[:, 0]] = pixel_vals

    return new_im, reference_coords



im = skio.imread("../original_images/test.jpg") / 255
corr1 = np.array([[0, 0], [1233, 0.], [1046, 307], [422, 307]])
# corr2 = np.array([[0, 0], [624, 0], [624, 294.24920128], [0, 294.24920128]])
corr2 = np.array([[0, 0], [624, 0], [624, 294], [0, 294]])

new_im, reference_coords = rectify(im, corr1, (624, 294))

print(reference_coords.shape)

plt.imshow(im)
# plt.scatter(reference_coords[:, 0], reference_coords[:, 1])
plt.show()

new_im1 = np.zeros((624, 294, 3))
rect_vertices = np.array(get_corners(new_im1))

plt.imshow(new_im)
# plt.scatter(pixel_coords[:, 0], pixel_coords[:, 1])
plt.show()