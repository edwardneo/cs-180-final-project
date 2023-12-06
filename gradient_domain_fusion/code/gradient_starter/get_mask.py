import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage.measure import find_contours

# from mask2chain import mask2chain_tmp

def get_mask(im):
    def on_key(event):
        if event.key == 'q':
            plt.close()

    print('Draw polygon around source object in clockwise order, q to stop')
    fig = plt.figure()
    plt.imshow(im)
    plt.axis('image')
    fig.canvas.mpl_connect('key_press_event', on_key)

    sx = []
    sy = []

    while True:
        points = plt.ginput(-1, timeout=0)
        if not point:
            plt.clf()
            plt.close()
            break
        x, y = point[0]
        sx.append(x)
        sy.append(y)
        plt.plot(sx, sy, '*-')
        plt.draw()
    
    print('h')

    mask = np.array(polygon(np.array(sy), np.array(sx), shape=im.shape[:2]))

    print('f')
    poly = None
    # if len(sx) > 0 and len(sy) > 0:
    #     contours = find_contours(mask, 0.5, positive_orientation='low')
    #     poly = {'x': contours[0][:, 1], 'y': contours[0][:, 0]}

    return mask, poly