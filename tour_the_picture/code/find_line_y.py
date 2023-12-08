def find_line_y(vx, vy, x1, y1, x2):
    # Function to find y-coordinate on the line passing through (vx, vy) and (y1, x1) at x2
    if vx == x1:
        return y1
    return vy + (x2 - vx) * (y1 - vy) / (x1 - vx)