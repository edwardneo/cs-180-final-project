import numpy as np
import skimage as sk
from scipy.interpolate import RegularGridInterpolator


def homog(pts):
    pts = np.array(pts)
    assert pts.shape[1] == 2, f"Points are of shape {pts.shape}"

    return np.vstack((pts.T, np.ones((1, pts.shape[0]))))


def inv_homog(pts):
    pts = np.array(pts)
    assert pts.shape[0] == 3, f"Points are of shape {pts.shape}"
    assert np.all(
        pts[2, :] == 1
    ), f"Points are not homogenous with final row {pts[2, :]}"

    return pts[:2, :].T


def normalize_homog(pts):
    pts = np.array(pts)
    assert pts.shape[0] == 3, f"Points are of shape {pts.shape}"

    return pts / pts[2, :]


def get_corners(im):
    height, width, _ = im.shape
    return np.array([(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)])


def get_range(corners):
    return (int(np.min(corners[:, 1])), int(np.max(corners[:, 1])) + 1), (
        int(np.min(corners[:, 0])),
        int(np.max(corners[:, 0])) + 1,
    )


def get_translation(range):
    return np.array((range[1][0], range[0][0]))


def get_bounding_box(range):
    return np.zeros((range[0][1] - range[0][0], range[1][1] - range[1][0], 3))


def computeH(corr1, corr2):
    """Computes a homography from corr1 into corr2."""
    corr1, corr2 = np.array(corr1), np.array(corr2)

    assert (
        corr1.shape == corr2.shape
    ), f"Points do not match with shapes {corr1.shape} and {corr2.shape}"
    assert (
        corr1.shape[1] == corr2.shape[1] == 2
    ), f"Points are not 2D with shapes {corr1.shape} and {corr2.shape}"
    assert (
        corr1.ndim == corr2.ndim == 2
    ), f"Point arrays have too many dimensions with shapes {corr1.shape} and {corr2.shape}"

    N = corr1.shape[0]

    A_even_rows = np.hstack(
        (corr1, np.ones((N, 1)), np.zeros((N, 3)), -corr1 * corr2[:, 0][:, np.newaxis])
    )
    A_odd_rows = np.hstack(
        (np.zeros((N, 3)), corr1, np.ones((N, 1)), -corr1 * corr2[:, 1][:, np.newaxis])
    )

    A = np.hstack((A_even_rows, A_odd_rows)).reshape((-1, 8))
    b = corr2.flatten()

    H_vec = np.concatenate((np.linalg.lstsq(A, b, rcond=-1)[0], np.ones(1)))
    H = H_vec.reshape(3, 3)
    return H


def rectify(im, corr_im, rect_dim):
    """Rectifies an image im given the corners of the image corr_im and the dimensions of
    the rectangle to rectify into."""
    im_interp = RegularGridInterpolator(
        (np.arange(im.shape[1]), np.arange(im.shape[0])),
        np.transpose(im, (1, 0, 2)),
        bounds_error=False,
        fill_value=0.5,
    )

    new_im = np.zeros((rect_dim[1], rect_dim[0], 3))
    rect_vertices = np.array(get_corners(new_im))
    pixel_coords = np.array(sk.draw.polygon(rect_vertices[:, 0], rect_vertices[:, 1])).T
    H = computeH(rect_vertices, corr_im)
    reference_coords = inv_homog(normalize_homog(H @ homog(pixel_coords)))
    pixel_vals = im_interp(reference_coords)

    new_im[pixel_coords[:, 1], pixel_coords[:, 0]] = pixel_vals

    return new_im
