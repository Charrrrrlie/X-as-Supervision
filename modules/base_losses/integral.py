# define label
def generate_joint_location_label(patch_width, patch_height, _joints, _joints_vis, *args):
    joints = _joints.copy()
    joints_vis = _joints_vis.copy()
    joints[:, 0] = joints[:, 0] * 1.0 / patch_width - 0.5
    joints[:, 1] = joints[:, 1] * 1.0 / patch_height - 0.5
    joints[:, 2] = joints[:, 2] * 1.0 / patch_width  # NOTE(xiao): we assume depth == width
    joints = joints.reshape((-1))
    joints_vis = joints_vis.reshape((-1))
    return joints, joints_vis


def get_label_func():
    return generate_joint_location_label