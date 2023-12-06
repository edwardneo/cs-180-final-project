import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours

def mask2chain_tmp(mask):
    crack_img = seg2cracks(mask)
    fragments = cracks2fragments(crack_img, mask, 1)
    x = np.round(fragments[0][:, 0]).astype(int)
    y = np.round(fragments[0][:, 1]).astype(int)

    gy, gx = np.gradient(mask.astype(float))
    epix = (gy**2 + gx**2 > 0) & mask
    e = y + (x-1)*crack_img.shape[0]
    ind = epix[e]
    mean_ind = np.mean(ind)
    x = x[ind]
    y = y[ind]

    plt.figure(1)
    plt.clf()
    plt.plot(x, y)
    plt.axis('image')

def cracks2fragments(crack_img, seg, isolation_check):
    if isolation_check is None:
        isolation_check = True

    nrows, ncols = crack_img.shape
    JUNCTION_BIT = 5

    onborder = np.zeros((nrows, ncols), dtype=bool)
    onborder[:, [0, -1]] = True
    onborder[[0, -1], :] = True

    num_neighbors_lookup = np.full((nrows, ncols), 4)
    num_neighbors_lookup[[0, -1], :] = 3
    num_neighbors_lookup[:, [0, -1]] = 3
    num_neighbors_lookup[[0, nrows-1, -nrows+1, -1]] = 2

    neighbor_offsets = np.array([-1, 1, -nrows, nrows])

    check_bit_interior = np.array([2, 4, 3, 1])

    valid_border_neighbors = np.empty((nrows, ncols), dtype=object)
    valid_border_neighbors[1:-1, 1:-1] = [2, 4, 3, 1]
    valid_border_neighbors[[0, -1], 1:-1] = [4, 3, 1]
    valid_border_neighbors[1:-1, [0, -1]] = [2, 4, 3]
    valid_border_neighbors[[0, -1], [0, -1]] = [4, 3]

    junction_map = np.bitwise_and(np.bitwise_or(crack_img == 11, crack_img == 7),
                                   np.bitwise_or(np.bitwise_or(crack_img == 14, crack_img == 13),
                                                 np.bitwise_or(crack_img == 15, crack_img == 13)))

    junction_index = np.where(junction_map)
    num_junctions = len(junction_index[0])

    junction_map = np.zeros_like(junction_map, dtype=int)
    junction_map[junction_index] = np.arange(1, num_junctions + 1)
    junction_fragmentlist = [[] for _ in range(num_junctions)]
    fragment_junctionlist = []
    fragment_segments = [None] * np.max(seg)

    junction_y, junction_x = np.unravel_index(junction_index, (nrows, ncols))
    junctions = np.column_stack((junction_x + 0.5, junction_y + 0.5))

    not_in_fragment = np.ones((nrows, ncols), dtype=bool)

    fragment_indices = np.zeros(nrows * ncols, dtype=int)
    fragment_ctr = 0
    fragments = []

    for i in range(num_junctions):
        if not onborder[junction_index[0][i]]:
            junction_neighbors = junction_index[0][i] + neighbor_offsets
            check_bit = check_bit_interior
        else:
            junction_neighbors = junction_index[0][i] + neighbor_offsets[valid_border_neighbors[junction_index[0][i]]]
            check_bit = check_bit_interior[valid_border_neighbors[junction_index[0][i]]]

        neighbor_bits = np.bitwise_and(crack_img[junction_neighbors], 1 << (check_bit - 1))
        which_junction_neighbors = np.where(np.bitwise_and(neighbor_bits, not_in_fragment[junction_neighbors]))

        which_junction_neighbors = which_junction_neighbors[0]

        which_junction_neighbors = which_junction_neighbors[
            np.where((junction_map[junction_neighbors[which_junction_neighbors]] == 0) |
                     (junction_neighbors[which_junction_neighbors] >= junction_index[0][i]))]

        for j in range(len(which_junction_neighbors)):
            if not not_in_fragment[junction_neighbors[which_junction_neighbors[j]]]:
                continue

            not_in_fragment[junction_index[0][i]] = False

            fragment_ctr += 1
            junction_fragmentlist[i].append(fragment_ctr)
            fragment_junctionlist.append(i + 1)

            fragment_length = 1
            fragment_indices[0] = junction_index[0][i]

            if not onborder[junction_index[0][i]]:
                step_dir = which_junction_neighbors[j]
            else:
                step_dir = valid_border_neighbors[junction_index[0][i]][which_junction_neighbors[j]]

            if step_dir == 1:
                fragment_segments.append([seg[junction_index[0][i]], seg[junction_index[0][i] + nrows]])
            elif step_dir == 2:
                fragment_segments.append([seg[junction_index[0][i] + 1 + nrows], seg[junction_index[0][i] + 1]])
            elif step_dir == 3:
                fragment_segments.append([seg[junction_index[0][i] + 1], seg[junction_index[0][i]]])
            elif step_dir == 4:
                fragment_segments.append([seg[junction_index[0][i] + nrows], seg[junction_index[0][i] + 1 + nrows]])
            else:
                raise ValueError('Invalid neighbor chosen???')

            neighbors = junction_neighbors
            which_neighbors = which_junction_neighbors[j]

            while which_neighbors.size > 0:
                fragment_length += 1
                fragment_indices[fragment_length - 1] = neighbors[which_neighbors]

                if fragment_length == 3:
                    not_in_fragment[junction_index[0][i]] = True

                if junction_map[neighbors[which_neighbors]]:
                    if neighbors[which_neighbors] != junction_index[0][i]:
                        junction_fragmentlist[junction_map[neighbors[which_neighbors]] - 1].append(fragment_ctr)
                    fragment_junctionlist[fragment_ctr - 1].append(junction_map[neighbors[which_neighbors]])

                    break

                not_in_fragment[fragment_indices[fragment_length - 1]] = False

                crnt = fragment_indices[fragment_length - 1]
                if not onborder[crnt]:
                    neighbors = crnt + neighbor_offsets
                    check_bit = check_bit_interior
                else:
                    neighbors = crnt + neighbor_offsets[valid_border_neighbors[crnt]]
                    check_bit = check_bit_interior[valid_border_neighbors[crnt]]

                neighbor_bits = np.bitwise_and(crack_img[neighbors], 1 << (check_bit - 1))
                which_neighbors = np.where(np.bitwise_and(neighbor_bits, not_in_fragment[neighbors]))

