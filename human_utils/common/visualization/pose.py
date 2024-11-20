import cv2
import numpy as np


def cv_draw_joints(im, kpt, vis, flip_pair_ids, color_left=(106,90,205), color_right=(255,165,0), radius=4):
    for ipt in range(0, kpt.shape[0]):
        if vis[ipt, 0]:
            cv2.circle(im, (int(kpt[ipt, 0] + 0.5), int(kpt[ipt, 1] + 0.5)), radius, color_left, -1)
    
    if flip_pair_ids is not None:
        for i in range(0, flip_pair_ids.shape[0]):
            id = flip_pair_ids[i][0]
            if vis[id, 0]:
                cv2.circle(im, (int(kpt[id, 0] + 0.5), int(kpt[id, 1] + 0.5)), radius, color_right, -1)


def cv_draw_joints_parent(im, kpt, vis, parent_ids, color=(192,192,192)):
    if kpt.shape[0] == len(parent_ids):
        for i in range(0, len(parent_ids)):
            id = parent_ids[i]
            if vis[id, 0] and vis[i, 0]:
                cv2.line(im, (int(kpt[i, 0] + 0.5), int(kpt[i, 1] + 0.5)), (int(kpt[id, 0] + 0.5), int(kpt[id, 1] + 0.5)),
                        color, thickness=2)


def plot_3d_skeleton(ax, kpt_3d, parent_ids, flip_pair_ids, patch_width, patch_height, c0='r',
                     c1='b', c2='g'):
    # joints
    X = kpt_3d[:, 0]
    Y = kpt_3d[:, 1]
    Z = kpt_3d[:, 2]

    for i in range(0, kpt_3d.shape[0]):
        if i == 0:
            ax.scatter(X[i], Z[i], -Y[i], c=c0, marker='*')
        else:
            ax.scatter(X[i], Z[i], -Y[i], c=c0, marker='o')
        # ax.scatter(-X[i], Z[i], -Y[i], c=c0, marker='o')
        if parent_ids is not None and len(kpt_3d) == len(parent_ids):
            x = np.array([X[i], X[parent_ids[i]]], dtype=np.float32)
            y = np.array([Y[i], Y[parent_ids[i]]], dtype=np.float32)
            z = np.array([Z[i], Z[parent_ids[i]]], dtype=np.float32)

            c = c1  # 'b'
            for j in range(0, flip_pair_ids.shape[0]):
                if i == flip_pair_ids[j][0]:
                    c = c2  # 'g'
                    break
            ax.plot(x, z, -y, c=c)
        # ax.plot(-x, z, -y, c=c)

    ax.set_aspect('equal')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Z Label')
    # ax.set_zlabel('Y Label')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.w_xaxis.line.set_visible(False)
    ax.w_yaxis.line.set_visible(False)
    ax.w_zaxis.line.set_visible(False)
    ax.grid(True)
    ax.view_init(elev=130, azim=-90)