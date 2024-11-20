import numpy as np
import skfmm

def compute_centroid(mask):
    _, h, w = mask.shape
    grid = np.mgrid[0:h, 0:w]
    center = np.array([np.sum(grid[1] * mask) / np.sum(mask), np.sum(grid[0] * mask) / np.sum(mask)]).astype(np.int16)

    # head = np.where(mask)
    # return np.array([center, [head[2][0], head[1][0]]])

    return center

def compute_geodesic_dis(img, img_path, geodesic_param_list, centers=None, is_norm=True):
    mask = np.bool_(img)
    if centers is None:
        centers = compute_centroid(mask).reshape(-1, 2)
    else:
        centers = centers.copy().astype(np.int16)

    DEBUG = False
    for center in centers:
        if img[0, center[1], center[0]] == 0 or DEBUG:
            return np.ones_like(img).astype(np.float16), centers

    mask_ = ~mask
    m = np.ones_like(img)
    for center in centers:
        m[0, center[1], center[0]] = 0
    m = np.ma.masked_array(m, mask_)

    distance = np.array(skfmm.distance(m))

    m_bg = np.ones_like(img)
    m_bg[mask] = geodesic_param_list[4]
    distance_bg = np.array(skfmm.distance(m_bg))

    if np.isnan(distance_bg).any() or np.isinf(distance_bg).any() or np.max(distance_bg) < 1:
        # check mask
        print(img_path)

    if is_norm:
        distance = distance / np.max(distance)
        distance = np.exp(geodesic_param_list[0] * distance)
        # distance = np.clip(distance, 0, 0.6 * distance.max())
        distance = distance + geodesic_param_list[1]

        distance_bg = distance_bg / np.max(distance_bg)
        distance_bg = geodesic_param_list[2] * distance_bg
        distance_bg = distance_bg + geodesic_param_list[3]

    # distance = distance + distance_bg * mask_
    distance = distance + distance_bg

    return distance, centers