"""
General image database
"""

import os
from easydict import EasyDict as edict


def patch_sample_full(image, center_x, center_y, width, height, rot, joints_3d, joints_3d_vis, flip_pairs, parent_ids):
    s = edict()
    s.image = image
    s.center_x = center_x
    s.center_y = center_y
    s.width = width
    s.height = height
    s.rot = rot
    s.joints_3d = joints_3d
    s.joints_3d_vis = joints_3d_vis
    s.flip_pairs = flip_pairs
    s.parent_ids = parent_ids
    return s


def patch_track_sample_full(image, center_x, center_y, width, height, rot, joints_3d, joints_3d_vis,
                            flip_pairs, parent_ids,
                            image_pre, center_pre_x, center_pre_y, width_pre, height_pre, joints_3d_pre,
                            joints_3d_vis_pre):
    s = patch_sample_full(image, center_x, center_y, width, height, rot, joints_3d, joints_3d_vis, flip_pairs,
                          parent_ids)
    s.image_pre = image_pre
    s.center_pre_x = center_pre_x
    s.center_pre_y = center_pre_y
    s.width_pre = width_pre
    s.height_pre = height_pre
    s.joints_3d_pre = joints_3d_pre
    s.joints_3d_vis_pre = joints_3d_vis_pre
    return s


def patch_sample_image(image, center_x, center_y, width, height):
    s = edict()
    s.image = image
    s.center_x = center_x
    s.center_y = center_y
    s.width = width
    s.height = height
    s.rot = None
    s.joints_3d = None
    s.joints_3d_vis = None
    s.flip_pairs = None
    s.parent_ids = None
    return s


def patch_sample_empty():
    s = edict()
    s.image = None
    s.center_x = None
    s.center_y = None
    s.width = None
    s.height = None
    s.rot = None
    s.joints_3d = None
    s.joints_3d_vis = None
    s.flip_pairs = None
    s.parent_ids = None
    return s


def patch_track_sample_empty():
    s = patch_sample_empty()
    s.image_pre = None
    s.center_pre_x = None
    s.center_pre_y = None
    s.width_pre = None
    s.height_pre = None
    s.joints_3d_pre = None
    s.joints_3d_vis_pre = None
    return s


def patch_sample(*args):
    # and isinstance(args[0], str)
    if len(args) == 0:
        return patch_sample_empty()
    elif len(args) == 5:
        return patch_sample_image(args[0], args[1], args[2], args[3], args[4])
    elif len(args) == 10:
        return patch_sample_full(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8],
                                 args[9])
    else:
        assert 0, "Error, invalid input argument number!"


def patch_track_sample(*args):
    if len(args) == 0:
        return patch_sample_empty()
    elif len(args) == 17:
        return patch_track_sample_full(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8],
                                       args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16])
    else:
        assert 0, "Error, invalid input argument number!"


class IMDB(object):
    def __init__(self, benchmark_name, image_set_name, dataset_path, patch_width, patch_height,
                 cache_path_root, extra_param):
        """
        basic information about an image database
        :param benchmark_name: name of benchmark
        :param image_set_name: name of image subset
        :param dataset_path: dataset path store images
        :param patch_width:
        :param patch_height:
        :param cache_path: store cache
        """
        self.benchmark_name = benchmark_name
        self.image_set_name = image_set_name
        self.dataset_path = dataset_path
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.cache_path_root = cache_path_root
        self.num_images = 0
        self.name = benchmark_name + '_' + image_set_name + '_w{}xh{}'.format(patch_width, patch_height) + extra_param

    @property
    def cache_path(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        cache_path = os.path.join(self.cache_path_root, '{}_cache'.format(self.name))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path
