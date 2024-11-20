import torch
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from train_util import pose_vis

def switch_points(points, gt, switch_all=False, \
                  switch_list=[(1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13)]):
    points_trans = points.clone()
    permutation = list(range((points.shape[1])))
    for a, b in switch_list:
        permutation[a] = b
        permutation[b] = a
    points_trans = points_trans[:, permutation, :]

    error_trans = torch.abs(points_trans - gt)[..., :2]
    error = torch.abs(points - gt)[..., :2]

    if switch_all:
        error_trans = error_trans.sum(dim=[1,2], keepdim=True)
        error = error.sum(dim=[1,2], keepdim=True)
    else:
        error_trans = error_trans.sum(dim=[2], keepdim=True)
        error = error.sum(dim=[2], keepdim=True)

    is_trans = torch.lt(error_trans, error)
    res = torch.where(is_trans, points_trans, points)

    return res, is_trans

def per_act_mse(pred, gt):

    pred = (pred.clone() + 1) / 2
    gt = (gt.clone() + 1) / 2

    error = pred - gt
    error = (error ** 2).sum(dim=2)
    error = torch.sqrt(error).mean(dim=1)   # (BN2)

    return error

def cal_per_class_error_(record_table, count_table):
    full_err = 0.0
    select_err = 0.0

    for k in record_table.keys():
        record_table[k] /= (count_table[k] + 1e-8)
        full_err += record_table[k]
        if k in ['Waiting', 'Posing', 'Greeting', 'Directions', 'Discussion', 'Walking']:
            select_err += record_table[k]

    full_err /= len(record_table)
    select_err /= 6

    return full_err, select_err

def cal_per_class_error(record_table, count_table, multi=False):
    if multi:
        full_err = {}
        select_err = {}
        for metric in record_table.keys():
            full_err[metric], select_err[metric] = cal_per_class_error_(record_table[metric], count_table[metric])
    else:
        full_err, select_err = cal_per_class_error_(record_table, count_table)
    return full_err, select_err


# paritally from https://github.com/una-dinosauria/3d-pose-baseline
def show3Dpose(vals, ax, lcolor="#3498db", rcolor="#F0E68C", add_labels=False, RADIUS=500): # blue, orange
    """
    Visualize a 3d skeleton

    Args
    channels: 96x1 vector. The pose to plot.
    ax: matplotlib 3d axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
    Returns
    Nothing. Draws on ax.
    """

    I   = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]) # start points
    J   = np.array([0, 1, 2, 0, 4, 5, 0, 17, 8, 9, 17, 11, 12, 17, 14, 15, 7]) # end points
    I   = np.array([1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17]) # start points
    J   = np.array([0, 1, 2, 0, 4, 5, 0, 17, 17, 11, 12, 17, 14, 15, 7]) # end points
    LR = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0 ,0], dtype=bool)

    # Make connection matrix
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=5, c=lcolor if LR[i] else rcolor)

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    # Get rid of the ticks and tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.set_zticklabels([])

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Keep z pane

    # set grid equal
    ax.set_aspect('equal')

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)


def show2Dpose(vals, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
    """Visualize a 2d skeleton

    Args
    channels: 64x1 vector. The pose to plot.
    ax: matplotlib axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
    Returns
    Nothing. Draws on ax.
    """

    I   = np.array([1, 2, 3, 4, 5, 6, 7, 8,  9, 10,11, 12, 13, 14, 15, 16, 17]) # start points
    J   = np.array([0, 1, 2, 0, 4, 5, 0, 17, 8, 9, 17, 11, 12, 17, 14, 15, 7]) # end points
    LR = np.array([0, 0, 0, 1, 1, 1, 0, 0,  0, 0, 1, 1, 1, 0, 0, 0 ,0], dtype=bool)

    # Make connection matrix
    for i in np.arange( len(I) ):
        x, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(2)]
        ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Get rid of tick labels
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    RADIUS = 350 # space around the subject
    xroot, yroot = vals[0,0], vals[0,1]
    ax.set_xlim([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim([-RADIUS+yroot, RADIUS+yroot])
    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("z")

    ax.set_aspect('equal')

def draw(p2d_front, front_img, p2d_back, back_img, p3d, p3d_gt, output_path, flip_pairs, parent_ids):
    # 1080p	= 1,920 x 1,080
    fig = plt.figure(figsize=(19.2, 10.8))

    gs1 = gridspec.GridSpec(1, 4)
    gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
    plt.axis('off')

    ax1 = plt.subplot(gs1[0])
    img = pose_vis(p2d_front, (256,256), flip_pairs, img=front_img, parent_ids=parent_ids)
    img = img.transpose(1,2,0)
    ax1.imshow(img)
    # ax1.invert_yaxis()
    plt.axis('off')

    ax2 = plt.subplot(gs1[1])
    img = pose_vis(p2d_back, (256,256), flip_pairs, img=back_img, parent_ids=parent_ids)
    img = img.transpose(1,2,0)
    ax2.imshow(img)
    # ax2.invert_yaxis()
    plt.axis('off')

    ax3 = plt.subplot(gs1[2], projection='3d')
    show3Dpose(p3d, ax3, lcolor="#6A5ACD", rcolor="#FFA500")

    ax4 = plt.subplot(gs1[3], projection='3d')
    show3Dpose(p3d_gt, ax4, lcolor="#3498db", rcolor="#F0E68C")

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def draw_2d(p2d_front, front_img, p2d_back, back_img, output_path, flip_pairs, parent_ids):
    # 1080p	= 1,920 x 1,080
    fig = plt.figure(figsize=(19.2, 10.8))

    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
    plt.axis('off')

    ax1 = plt.subplot(gs1[0])
    img = pose_vis(p2d_front, (256,256), flip_pairs, img=front_img, parent_ids=parent_ids)
    img = img.transpose(1,2,0)
    ax1.imshow(img)
    # ax1.invert_yaxis()
    plt.axis('off')

    ax2 = plt.subplot(gs1[1])
    img = pose_vis(p2d_back, (256,256), flip_pairs, img=back_img, parent_ids=parent_ids)
    img = img.transpose(1,2,0)
    ax2.imshow(img)
    plt.axis('off')

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def draw_mono(img, p2d, p3d, output_path, flip_pairs, parent_ids):
    fig = plt.figure(figsize=(19.2, 10.8))

    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
    plt.axis('off')

    ax0 = plt.subplot(gs1[0])
    img = img.transpose(1,2,0)
    ax0.imshow(img)
    plt.axis('off')

    ax2 = plt.subplot(gs1[1], projection='3d')
    show3Dpose(p3d, ax2, lcolor="#6A5ACD", rcolor="#FFA500", RADIUS=120)

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def draw_mono_2d(img, p2d,output_path, flip_pairs, parent_ids):
    fig = plt.figure(figsize=(19.2, 10.8))

    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
    plt.axis('off')

    ax0 = plt.subplot(gs1[0])
    img = img.transpose(1,2,0)
    ax0.imshow(img)
    # ax1.invert_yaxis()
    plt.axis('off')

    ax1 = plt.subplot(gs1[1])
    img = pose_vis(p2d, (256,256), flip_pairs, img=img, parent_ids=parent_ids)
    img = img.transpose(1,2,0)
    ax1.imshow(img)
    plt.axis('off')

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()