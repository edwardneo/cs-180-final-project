import numpy as np
import matplotlib.pyplot as plt

def align_source(im_s, mask, im_t):
    plt.figure(1)
    plt.clf()
    plt.imshow(im_s)
    plt.axis('image')

    plt.figure(2)
    plt.clf()
    plt.imshow(im_t)
    plt.axis('image')

    y, x = np.where(mask)
    y1, y2 = min(y)-1, max(y)+1
    x1, x2 = min(x)-1, max(x)+1

    im_s2 = np.zeros_like(im_t)

    print('Choose target bottom-center location')
    tx, ty = plt.ginput(1)[0]

    yind = np.arange(y1, y2+1)
    yind2 = yind - max(y) + round(ty)
    xind = np.arange(x1, x2+1)
    xind2 = xind - round(np.mean(x)) + round(tx)

    y = y - max(y) + round(ty)
    x = x - round(np.mean(x)) + round(tx)
    ind = y + (x - 1) * im_t.shape[0]
    mask2 = np.zeros_like(im_t, dtype=bool)
    mask2.flat[ind] = True

    im_s2[yind2, xind2, :] = im_s[yind, xind, :]
    im_t[np.tile(mask2[:, :, None], (1, 1, 3))] = im_s2[np.tile(mask2[:, :, None], (1, 1, 3))]

    plt.figure(1)
    plt.clf()
    plt.imshow(im_s2)
    plt.axis('image')
    
    plt.figure(2)
    plt.clf()
    plt.imshow(im_t)
    plt.axis('image')
    
    plt.draw()
    plt.show()

    return im_s2, mask2