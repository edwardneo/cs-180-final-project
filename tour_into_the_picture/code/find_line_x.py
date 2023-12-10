def find_line_x(vx, vy, x1, y1, y2):
    """Finds the x-coordinate on the line passing through (vx, vy) and (x1, y1) at y2"""
    if vy == y1:
        return x1
    return vx + (y2 - vy) * (x1 - vx) / (y1 - vy)