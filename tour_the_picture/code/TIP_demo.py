import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage.io as skio

from TIP_GUI import tip_gui
from TIP_get5rects import tip_get5rects
from warp import rectify


def dist2edges(point, upper_left, bottom_right):
    """Gets distances from point to edges of bounding box
    defined by upper_left and bottom_right"""

    assert (
        upper_left[0] <= point[0] <= bottom_right[0]
        and upper_left[1] <= point[1] <= bottom_right[1]
    ), "Point not within bounding box"
    return (
        point[1] - upper_left[1],
        bottom_right[0] - point[0],
        bottom_right[1] - point[1],
        point[0] - upper_left[0],
    )


def get_bounding_box(rx, ry):
    """Gets upper left and bottom right corners given x and y
    coordinates of the corners ordered clockwise starting from
    the top right corner"""

    return (rx[0], ry[0]), (rx[2], ry[2])

def get_corners(top_left, bottom_right):
    """Helper function to get corners of a rectangle from the top left and bottom right
    corners."""

    return (np.array([top_left[0], bottom_right[0], bottom_right[0], top_left[0], top_left[0]]),
        np.array([top_left[1], top_left[1], bottom_right[1], bottom_right[1], top_left[1]]))


# Function to display the expended image
def display_expanded_image(bim, bim_alpha, vx, vy, rects):
    # Display the expended image
    plt.figure()
    plt.imshow(bim)

    # Draw the Vanishing Point and the 4 faces on the image
    plt.plot(vx, vy, "w*")
    for rect in rects:
        plt.plot(
            [rect[0], rect[1], rect[2], rect[3], rect[0]],
            [rect[4], rect[5], rect[6], rect[7], rect[4]],
            "y-",
        )

    plt.show()


# Function to display 3D surfaces
def display_3d_surfaces(bim, bim_alpha):
    # Define a surface in 3D
    planex = np.array([[0, 0, 0], [0, 0, 0]])
    planey = np.array([[-1, 0, 1], [-1, 0, 1]])
    planez = np.array([[1, 1, 1], [0, 0, 0]])

    # Create the surface and texture map it with a given image
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        planex,
        planey,
        planez,
        facecolors=plt.imshow(bim),
        alpha=0.5,
        rstride=5,
        cstride=5,
    )

    # Alpha-channel magic to make things transparent
    ax.alpha = bim_alpha
    ax.alpha("texture")

    # Some 3D magic...
    ax.axis("equal")  # Make X, Y, Z dimensions be equal
    ax.axis("vis3d")  # Freeze the scale for better rotations
    ax.axis("off")  # Turn off the tick marks

    # Make it a perspective projection
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 1, 1]))

    # Use the "rotate 3D" button on the figure or do "View->Camera Toolbar"
    # to rotate the figure
    # or use functions campos and camtarget to set camera location
    # and viewpoint from within Python code

    plt.show()

# Read in sample image
im = skio.imread("../original_images/sjerome.jpg") / 255

# Run the GUI
vx, vy, irx, iry, orx, ory = tip_gui(im)

# Find the cube faces and compute the expanded image
bim, bim_alpha, vx, vy, ceil_rx, ceil_ry, floor_rx, floor_ry, left_rx, left_ry, right_rx, right_ry, back_rx, back_ry = tip_get5rects(im, vx, vy, irx, iry, orx, ory)

skio.imsave(f'../original_images/test.jpg', (bim * 255).astype(np.uint8))

# Display the expanded image
plt.clf()
plt.imshow(bim)

# Draw the vanishing Point and the 4 faces on the image
plt.plot(irx, iry, 'b')
plt.plot([vx, irx[0]], [vy, iry[0]], 'r-.')
plt.plot(ceil_rx, ceil_ry)
# for ix, iy, ox, oy in zip(irx, iry, orx, ory):
#     plt.plot([ox, ix], [oy, iy], 'r')
plt.pause(0.05)
plt.show()

# Get depths
f = 13.4e-3
f = 300

ir_upper_left, ir_bottom_right = get_bounding_box(irx, iry)
im_upper_left, im_bottom_right = (0, 0), (im.shape[1], im.shape[0])

ir_dists = dist2edges((vx, vy), ir_upper_left, ir_bottom_right)
im_dists = dist2edges((vx, vy), im_upper_left, im_bottom_right)

depths = np.zeros(4)

for i in range(4):
    depths[i] = f * (im_dists[i] / ir_dists[i] - 1)

# Rectify faces
back_width, back_height = np.max(back_rx) - np.min(back_rx), np.max(back_ry) - np.min(back_ry)
ceiling = rectify(im, np.stack((ceil_rx, ceil_ry)).T, (back_width, depths[0]))

plt.clf()
plt.imshow(ceiling)
plt.pause(0.05)
plt.show()


# # # Display the expended image
# # display_expanded_image(im, None, vx, vy, rects)

# # # Display 3D surfaces
# # display_3d_surfaces(im, None)
