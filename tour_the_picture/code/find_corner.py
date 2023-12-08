from find_line_x import find_line_x
from find_line_y import find_line_y

def find_corner(vx, vy, rx, ry, limitx, limity):
    x1, y1 = find_line_x(vx, vy, rx, ry, limity), limity
    x2, y2 = limitx, find_line_y(vx, vy, rx, ry, limitx)

    if (vx - x1) ** 2 + (vy - y1) ** 2 > (vx - x2) ** 2 + (vy - y2) ** 2:
        x, y = x1, y1
    else:
        x, y = x2, y2

    return x, y