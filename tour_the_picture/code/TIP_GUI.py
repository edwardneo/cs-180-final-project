import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import skimage.io as skio

from find_corner import find_corner

def get_corners(top_left, bottom_right):
    """Helper function to get corners of a rectangle from the top left and bottom right
    corners."""

    return (np.array([top_left[0], bottom_right[0], bottom_right[0], top_left[0], top_left[0]]),
        np.array([top_left[1], top_left[1], bottom_right[1], bottom_right[1], top_left[1]]))

def tip_gui(im):
    """GUI to get vanishing point, inner rectangle, and outer polygon. First left click
    twice to select the top right and bottom left corners. Then left click to choose a
    vanishing point. You may change the vanishing point by left clicking again. Right
    click when you have selected a satisfactory vanishing point and to close the GUI."""

    ymax, xmax, _ = im.shape
    plt.imshow(im)

    # Get the upper left and lower right corner of the inner rectangle
    top_left, bottom_right = plt.ginput(2, timeout=0)

    # Draw the rectangle
    irx, iry = get_corners(top_left, bottom_right)
    plt.plot(irx, iry, 'b')
    plt.pause(0.05)

    # Get the vanishing point
    v = []
    
    # Loop to require vanishing point
    while not v:
        v = plt.ginput(1, timeout=0, mouse_pop=None, mouse_stop=MouseButton.RIGHT)

    # Loop to get the user to specify the vanishing point
    while v:
        vx, vy = v[0]

        # Find where the line from VP thru inner rectangle hits the edge of the image
        im_corners_x, im_corners_y = get_corners((0, 0), (xmax, ymax))
        opx = np.zeros(4)
        opy = np.zeros(4)

        for i in range(4):
            opx[i], opy[i] = find_corner(vx, vy, irx[i], iry[i], im_corners_x[i], im_corners_y[i])
        
        orx, ory = get_corners((np.min(opx), np.min(opy)), (np.max(opx), np.max(opy)))

        # Draw everything
        plt.clf()
        plt.imshow(im)
        plt.plot(irx, iry, 'b')
        plt.plot([vx, irx[0]], [vy, iry[0]], 'r-.')
        for ix, iy, ox, oy in zip(irx, iry, orx, ory):
            plt.plot([ox, ix], [oy, iy], 'r')
        plt.pause(0.05)

        v = plt.ginput(1, timeout=0, mouse_pop=None, mouse_stop=MouseButton.RIGHT)

    return np.rint(vx), np.rint(vy), np.rint(irx), np.rint(iry), np.rint(orx), np.rint(ory)