import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage.io as skio
import argparse

from TIP_GUI import tip_gui
from TIP_get5rects import tip_get5rects
from warp import rectify


def dist2edges(point, upper_left, bottom_right):
    """Helper function to get distances from point to edges of bounding box defined
    by upper_left and bottom_right."""

    assert (
        upper_left[0] <= point[0] <= bottom_right[0]
        and upper_left[1] <= point[1] <= bottom_right[1]
    ), f"Point not within bounding box {point, upper_left, bottom_right}"
    return (
        point[1] - upper_left[1],
        bottom_right[0] - point[0],
        bottom_right[1] - point[1],
        point[0] - upper_left[0],
    )


def get_bounding_box(rx, ry):
    """Helper function to get upper left and bottom right corners given x and y
    coordinates of the corners ordered clockwise starting from the top right
    corner"""

    return (rx[0], ry[0]), (rx[2], ry[2])


def get_corners(top_left, bottom_right):
    """Helper function to get corners of a rectangle from the top left and bottom right
    corners."""

    return (
        np.array(
            [top_left[0], bottom_right[0], bottom_right[0], top_left[0], top_left[0]]
        ),
        np.array(
            [top_left[1], top_left[1], bottom_right[1], bottom_right[1], top_left[1]]
        ),
    )


def display_image_surface(im, ax, x, y, z, sample_rate=2):
    """Plots image im with axes ax on a planar surface perpendicular to one of the axes.
    One of x, y, z is an int indicating where the plane intersects with the perpendicular
    axis, and the other two are tuples or lists of length 2 indicating where on the plane
    the image will be displayed."""

    if type(x) == int:
        X = np.atleast_2d(x)
        Y = np.arange(y[0], y[1])
        Z = np.arange(z[0], z[1])
        Y, Z = np.meshgrid(Y, Z)
    elif type(y) == int:
        X = np.arange(x[0], x[1])
        Y = np.atleast_2d(y)
        Z = np.arange(z[0], z[1])
        X, Z = np.meshgrid(X, Z)
    elif type(z) == int:
        X = np.arange(x[0], x[1])
        Y = np.arange(y[0], y[1])
        Z = np.atleast_2d(z)
        X, Y = np.meshgrid(X, Y)
    else:
        raise TypeError("Not a flat surface")

    ax.plot_surface(X, Y, Z, rstride=sample_rate, cstride=sample_rate, facecolors=im)

def main(im_fn, f, sample_rate):
    # Read in sample image
    im = skio.imread(im_fn) / 255

    # Run the GUI
    vx, vy, irx, iry, orx, ory = tip_gui(im)

    # Find the cube faces and compute the expanded image
    (
        bim,
        bim_alpha,
        vx,
        vy,
        ceil_rx,
        ceil_ry,
        floor_rx,
        floor_ry,
        left_rx,
        left_ry,
        right_rx,
        right_ry,
        back_rx,
        back_ry,
    ) = tip_get5rects(im, vx, vy, irx, iry, orx, ory)

    # Display the expanded image
    plt.clf()
    plt.imshow(bim)

    # Draw the vanishing Point and the 4 faces on the image
    plt.plot(irx, iry, "b")
    for ix, iy, ox, oy in zip(irx, iry, orx, ory):
        plt.plot([ox, ix], [oy, iy], "r")

    plt.fill(ceil_rx, ceil_ry, alpha=0.5)
    plt.fill(right_rx, right_ry, alpha=0.5)
    plt.fill(floor_rx, floor_ry, alpha=0.5)
    plt.fill(left_rx, left_ry, alpha=0.5)
    plt.fill(back_rx, back_ry, alpha=0.5)

    plt.pause(0.05)
    plt.show()

    # Get depths
    back_upper_left, back_bottom_right = get_bounding_box(back_rx, back_ry)
    bim_upper_left, bim_bottom_right = (0, 0), (bim.shape[1], bim.shape[0])

    ir_dists = dist2edges((vx, vy), back_upper_left, back_bottom_right)
    im_dists = dist2edges((vx, vy), bim_upper_left, bim_bottom_right)

    depths = np.zeros(4)

    for i in range(4):
        depths[i] = f * (im_dists[i] / ir_dists[i] - 1)

    depths = np.rint(depths).astype(int)

    # Rectify faces
    back_width, back_height = int(np.rint(np.max(back_rx))) - int(
        np.rint(np.min(back_rx))
    ), int(np.rint(np.max(back_ry))) - int(np.rint(np.min(back_ry)))
    ceiling = rectify(bim, np.stack((ceil_rx, ceil_ry)).T, (back_width, depths[0]))
    right = rectify(bim, np.stack((right_rx, right_ry)).T, (depths[1], back_height))
    floor = rectify(bim, np.stack((floor_rx, floor_ry)).T, (back_width, depths[2]))
    left = rectify(bim, np.stack((left_rx, left_ry)).T, (depths[3], back_height))
    back = bim[
        int(np.rint(np.min(back_ry))) : int(np.rint(np.max(back_ry))),
        int(np.rint(np.min(back_rx))) : int(np.rint(np.max(back_rx))),
    ]

    # Create flattened image for sanity
    flat_im = np.zeros(
        (depths[0] + back_height + depths[2], depths[3] + back_width + depths[1], 3)
    )
    flat_im[: depths[0], depths[3] : depths[3] + back_width] = ceiling
    flat_im[depths[0] : depths[0] + back_height, depths[3] + back_width :] = right
    flat_im[depths[0] + back_height :, depths[3] : depths[3] + back_width] = floor
    flat_im[depths[0] : depths[0] + back_height, : depths[3]] = left
    flat_im[depths[0] : depths[0] + back_height, depths[3] : depths[3] + back_width] = back

    plt.clf()
    plt.imshow(flat_im)
    plt.pause(0.05)
    plt.show()


    # Display 3D rendering
    plt.clf()
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection="3d")
    display_image_surface(
        np.flip(np.transpose(ceiling, (1, 0, 2)), axis=1),
        ax,
        [0, depths[0]],
        [0, back_width],
        back_height,
        sample_rate=sample_rate
    )
    display_image_surface(
        np.flip(right, axis=0), ax, [0, depths[1]], back_width, [0, back_height], sample_rate=sample_rate
    )
    display_image_surface(
        np.transpose(floor, (1, 0, 2)),
        ax,
        [0, depths[2]],
        [0, back_width],
        0,
        sample_rate=sample_rate
    )
    display_image_surface(
        np.flip(left, axis=(1, 0)), ax, [0, depths[3]], 0, [0, back_height], sample_rate=sample_rate
    )
    display_image_surface(np.flip(back, axis=0), ax, 0, [0, back_width], [0, back_height], sample_rate=sample_rate)
    ax.axis("off")

    plt.pause(0.05)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an image with a specified focal length.')

    parser.add_argument('image_filename', type=str, help='Path to the input image file')
    parser.add_argument('--focal_length', type=float, default=300, help='Focal length of the camera (default: 300)')
    parser.add_argument('--sample_rate', type=int, default=2, help='Sample rate to construct 3D render with (default: 2)')

    args = parser.parse_args()

    main(args.image_filename, args.focal_length, args.sample_rate)