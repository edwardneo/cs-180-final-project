import numpy as np
import cv2

from find_line_x import find_line_x
from find_line_y import find_line_y

def tip_get5rects(im, vx, vy, irx, iry, orx, ory):
    ymax, xmax, cdepth = im.shape
    lmargin = -int(np.min(orx))
    rmargin = int(np.max(orx) - xmax)
    tmargin = -int(np.min(ory))
    bmargin = int(np.max(ory) - ymax)

    big_im = np.zeros([ymax + tmargin + bmargin, xmax + lmargin + rmargin, cdepth])
    big_im_alpha = np.zeros([big_im.shape[0], big_im.shape[1]])
    big_im[tmargin:ymax + tmargin, lmargin:xmax + lmargin, :] = im
    big_im_alpha[tmargin:ymax + tmargin, lmargin:xmax + lmargin] = 1.0

    vx += lmargin
    vy += tmargin
    irx += lmargin
    iry += tmargin
    orx += lmargin
    ory += tmargin

    # Define the 5 rectangles

    # Ceiling
    ceil_rx = [orx[0], orx[1], irx[1], irx[0]]
    ceil_ry = [ory[0], ory[1], iry[1], iry[0]]
    if ceil_ry[0] < ceil_ry[1]:
        ceil_rx[0] = np.round(find_line_x(vx, vy, ceil_rx[0], ceil_ry[0], ceil_ry[1]))
        ceil_ry[0] = ceil_ry[1]
    else:
        ceil_rx[1] = np.round(find_line_x(vx, vy, ceil_rx[1], ceil_ry[1], ceil_ry[0]))
        ceil_ry[1] = ceil_ry[0]

    # Floor
    floor_rx = [irx[3], irx[2], orx[2], orx[3]]
    floor_ry = [iry[3], iry[2], ory[2], ory[3]]
    if floor_ry[2] > floor_ry[3]:
        floor_rx[2] = np.round(find_line_x(vx, vy, floor_rx[2], floor_ry[2], floor_ry[3]))
        floor_ry[2] = floor_ry[3]
    else:
        floor_rx[3] = np.round(find_line_x(vx, vy, floor_rx[3], floor_ry[3], floor_ry[2]))
        floor_ry[3] = floor_ry[2]

    # Left
    left_rx = [orx[0], irx[0], irx[3], orx[3]]
    left_ry = [ory[0], iry[0], iry[3], ory[3]]
    if left_rx[0] < left_rx[3]:
        left_ry[0] = np.round(find_line_y(vx, vy, left_rx[0], left_ry[0], left_rx[3]))
        left_rx[0] = left_rx[3]
    else:
        left_ry[3] = np.round(find_line_y(vx, vy, left_rx[3], left_ry[3], left_rx[0]))
        left_rx[3] = left_rx[0]

    # Right
    right_rx = [irx[1], orx[1], orx[2], irx[2]]
    right_ry = [iry[1], ory[1], ory[2], iry[2]]
    if right_rx[1] > right_rx[2]:
        right_ry[1] = np.round(find_line_y(vx, vy, right_rx[1], right_ry[1], right_rx[2]))
        right_rx[1] = right_rx[2]
    else:
        right_ry[2] = np.round(find_line_y(vx, vy, right_rx[2], right_ry[2], right_rx[1]))
        right_rx[2] = right_rx[1]

    back_rx, back_ry = irx, iry

    return big_im, big_im_alpha, vx, vy, ceil_rx, ceil_ry, floor_rx, floor_ry, left_rx, left_ry, right_rx, right_ry, back_rx, back_ry
