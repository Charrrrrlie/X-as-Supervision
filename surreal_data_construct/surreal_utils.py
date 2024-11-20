# The code is partly from 
# https://github.com/gulvarol/surreal/blob/master/datageneration/misc/smpl_relations/smpl_relations.py

import numpy as np
import transforms3d

import matplotlib.pyplot as plt

def joint_names():
    return ['Pelvis',
            'RHip',
            'RKnee',
            'RAnkle',
            'LHip',
            'LKnee',
            'LAnkle',
            'Torso',
            'Neck',
            'Nose',
            'Head',
            'LShoulder',
            'LElbow',
            'LWrist',
            'RShoulder',
            'RElbow',
            'RWrist',
            'Thorax',]


def draw_joints2D(joints2D, ax=None, kintree_table=None, with_text=True, color='g'):
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for i in range(1, kintree_table.shape[1]):
        j1 = kintree_table[0][i]
        j2 = kintree_table[1][i]
        ax.plot([joints2D[j1, 0], joints2D[j2, 0]],
                [joints2D[j1, 1], joints2D[j2, 1]],
                color=color, linestyle='-', linewidth=2, marker='o', markersize=5)
        if with_text:
            ax.text(joints2D[j2, 0],
                    joints2D[j2, 1],
                    s=joint_names()[j2],
                    color=color,
                    fontsize=8)


def rotateBody(RzBody, pelvisRotVec):
    angle = np.linalg.norm(pelvisRotVec)
    Rpelvis = transforms3d.axangles.axangle2mat(pelvisRotVec / angle, angle)
    globRotMat = np.dot(RzBody, Rpelvis)
    R90 = transforms3d.euler.euler2mat(np.pi / 2, 0, 0)
    globRotAx, globRotAngle = transforms3d.axangles.mat2axangle(np.dot(R90, globRotMat))
    globRotVec = globRotAx * globRotAngle
    return globRotVec


# Returns intrinsic camera matrix
# Parameters are hard-coded since all SURREAL images use the same.
def get_intrinsic(res_x_px, res_y_px):
    # These are set in Blender (datageneration/main_part1.py)
    # res_x_px = 320  # *scn.render.resolution_x
    # res_y_px = 240  # *scn.render.resolution_y
    f_mm = 60  # *cam_ob.data.lens
    sensor_w_mm = 32  # *cam_ob.data.sensor_width
    sensor_h_mm = sensor_w_mm * res_y_px / res_x_px  # *cam_ob.data.sensor_height (function of others)

    scale = 1  # *scn.render.resolution_percentage/100
    skew = 0  # only use rectangular pixels
    pixel_aspect_ratio = 1

    # From similar triangles:
    # sensor_width_in_mm / resolution_x_inx_pix = focal_length_x_in_mm / focal_length_x_in_pix
    fx_px = f_mm * res_x_px * scale / sensor_w_mm
    fy_px = f_mm * res_y_px * scale * pixel_aspect_ratio / sensor_h_mm

    # Center of the image
    u = res_x_px * scale / 2
    v = res_y_px * scale / 2

    # Intrinsic camera matrix
    K = np.array([[fx_px, skew, u], [0, fy_px, v], [0, 0, 1]])
    return K

# Returns extrinsic camera matrix
#   T : translation vector from Blender (*cam_ob.location)
#   RT: extrinsic computer vision camera matrix
#   Script based on https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
def get_extrinsic(T):
    # Take the first 3 columns of the matrix_world in Blender and transpose.
    # This is hard-coded since all images in SURREAL use the same.
    R_world2bcam = np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]).transpose()
    # *cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']),
    #                               (0., -1, 0., -1.0),
    #                               (-1., 0., 0., 0.),
    #                               (0.0, 0.0, 0.0, 1.0)))

    # Convert camera location to translation vector used in coordinate changes
    T_world2bcam = -1 * np.dot(R_world2bcam, T)

    # Following is needed to convert Blender camera to computer vision camera
    R_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = np.dot(R_bcam2cv, R_world2bcam)
    T_world2cv = np.dot(R_bcam2cv, T_world2bcam)

    # Put into 3x4 matrix
    RT = np.concatenate([R_world2cv, T_world2cv], axis=1)
    return RT, R_world2cv, T_world2cv


def project_vertices(points, intrinsic, extrinsic, centralize_joints=False):
    homo_coords = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1).transpose()
    proj_coords = np.dot(intrinsic, np.dot(extrinsic, homo_coords))

    proj_coords = proj_coords.transpose()
    proj_coords[:, :2] = proj_coords[:, :2] / proj_coords[:, [2]]
    if centralize_joints:
        proj_coords[:, 2] = proj_coords[:, 2] - proj_coords[0, 2]

    return proj_coords


def get_frame(cap, t=0):
    cap.set(propId=1, value=t)
    ret, frame = cap.read()
    return frame

def get_mask(mat, t=0):
    mask = mat['segm_{}'.format(t + 1)]

    mask[mask!=0] = 1.
    mask = mask[..., None]
    return mask

def filter_incorrect_cases(mask, keypoints):
    # output number for debugging
    count = 0
    for i in range(keypoints.shape[0]):
        if mask[int(keypoints[i, 1]), int(keypoints[i, 0]), :] == 0:
            count +=1
    if count > 4:
        return -1
    return count