To run the code, run through `code/main.ipynb`. Change the input file names from `samples/` to choose different images to blend. The interactive GUI will let you create a mask.

All of the main functions are contained in `code/utils.py`:
- `poly2mask(vertex_row_coords, vertex_col_coords, shape)`: Creates a binary mask from polygon vertex coordinates.
- `specify_bottom_center(img)`: GUI to specify target bottom-center location.
- `align_source(object_img, mask, background_img, bottom_center)`: Aligns the object and the background.
- `upper_left_background_rc(object_mask, bottom_center)`: Returns upper-left (row,col) coordinate in background image that corresponds to (0,0) in the object image
- `crop_object_img(object_img, object_mask)`: Gets the excess zero margins in the mask and crops it off the image and the mask.
- `get_combined_img(bg_img, object_img, object_mask, bg_ul)`: Combines the two images.
- `specify_mask(img)`: GUI to trace the polygon border around the mask.
- `get_mask(ys, xs, img)`: Gets the mask from the polygon vertex coordinates and displays it.