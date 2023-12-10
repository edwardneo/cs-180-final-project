To run the code, run `code/TIP_demo.py` with parameters `image_filename`, `focal_length`, and `sample_rate`. Focal length is default 300, and sample rate is default 2. For example,

`python TIP_demo.py ../original_images/sjerome.jpg`

The main functions are all documented:
- `find_corner.py` `find_corner(vx, vy, rx, ry, limitx, limity)`: Finds a 'corner' of the bigger image given the vanishing point, the rectangle corner, and the boundary limits.
- `find_line_x.py` `find_line_x(vx, vy, x1, y1, y2)`: Finds the x-coordinate on the line passing through (vx, vy) and (x1, y1) at y2.
- `find_line_y.py` `find_line_y(vx, vy, x1, y1, x2)`: Finds the y-coordinate on the line passing through (vx, vy) and (y1, x1) at x2.
- `TIP_get5rects.py` `tip_get5rects(im, vx, vy, irx, iry, orx, ory)`: Gets the coordinates of the five rectangles using the image, vanishing point, inner rectangle, and outer rectangle.
- `TIP_GUI.py` `tip_gui(im)`: GUI to get vanishing point, inner rectangle, and outer polygon. First left click twice to select the top left and bottom right corners. Then left click to choose a vanishing point. You may change the vanishing point by left clicking again. Right click when you have selected a satisfactory vanishing point and to close the GUI.
- `warp.py` `rectify(im, corr_im, rect_dim)`: Rectifies an image im given the corners of the image corr_im and the dimensions of the rectangle to rectify into.
- `TIP_demo.py` `dist2edges(point, upper_left, bottom_right)`: Helper function to get distances from point to edges of bounding box defined by upper_left and bottom_right.
- `TIP_demo.py` `display_image_surface(im, ax, x, y, z, sample_rate=2)`: Plots image im with axes ax on a planar surface perpendicular to one of the axes. One of x, y, z is an int indicating where the plane intersects with the perpendicular axis, and the other two are tuples or lists of length 2 indicating where on the plane the image will be displayed.