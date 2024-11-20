

def CamProj(x, y, z, fx, fy, cx, cy):
    cam_x = x / z * fx
    cam_x = cam_x + cx
    cam_y = y / z * fy
    cam_y = cam_y + cy

    return cam_x, cam_y


def CamBackProj(cam_x, cam_y, depth, fx, fy, u, v):
    x = (cam_x - u) / fx * depth
    y = (cam_y - v) / fy * depth

    return x, y, depth
