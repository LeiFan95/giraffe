import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

relations = [[0, 1], [0, 4], [1, 2], [2, 3], [4, 5], [5, 6], 
        [0, 7], [7, 8], [8, 11], [8, 14], [14, 15], 
        [15, 16], [11, 12], [12, 13], [10, 9], [9, 8]]

can_pose = np.array([[ 0., -0.18367347, -1.],
                    [ 0.62962963, -0.18367347, -1.],
                    [ 0.62962963, 0.37755102, -1.],
                    [ 0.62962963, 1., -1.],
                    [-0.62962963, -0.18367347, -1.],
                    [-0.62962963, 0.37755102, -1.],
                    [-0.62962963, 0.96938776, -1.],
                    [ 0., -0.42857143, -1.],
                    [ 0., -0.75510204, -1.],
                    [ 0., -0.90816327, 1.],
                    [ 0., -1., -1.],
                    [-0.62962963, -0.6122449, -1.],
                    [-0.81481481, -0.30612245, -1.],
                    [-1., -0.10204082, -1.],
                    [0.62962963, -0.6122449,  -1.],
                    [0.81481481, -0.30612245, -1.],
                    [1., -0.10204082, -1.]])

can_xs, can_ys, can_zs = can_pose[:, 0], can_pose[:, 2], -can_pose[:, 1]

def animate_pose(ax, k_3d, k_3d_diff):
    xs, ys, zs = k_3d[:, 0], k_3d[:, 1], k_3d[:, 2]
    delta_xs, delta_ys, delta_zs = k_3d_diff[:, 0], k_3d_diff[:, 1], k_3d_diff[:, 2]
    anni_xs, anni_ys, anni_zs = can_xs + delta_xs, can_ys + delta_ys, can_zs + delta_zs
    
    for rel in relations:
        idx0, idx1 = rel[0], rel[1]
        # ax.plot([xs[idx0], xs[idx1]], [ys[idx0], ys[idx1]], [zs[idx0], zs[idx1]], linewidth=1, label=r'$z=y=x$')
        ax.plot([anni_xs[idx0], anni_xs[idx1]], [anni_ys[idx0], anni_ys[idx1]], [anni_zs[idx0], anni_zs[idx1]], linewidth=1, label=r'$z=y=x$', color='r')
        ax.plot([can_xs[idx0], can_xs[idx1]], [can_ys[idx0], can_ys[idx1]], [can_zs[idx0], can_zs[idx1]], linewidth=1, label=r'$z=y=x$', color='g')
    # ax.scatter(xs, ys, zs)
    ax.scatter(anni_xs, anni_ys, anni_zs)
    ax.scatter(can_xs, can_ys, can_zs)
    return ax

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    major_locator = MultipleLocator(0.25)

    idx_txt = '/home/leifan/h36m-fetch/data/Human3.6/cropped/names.txt'
    dataset_folder = '/data/leifan/dataset/Human3.6/cropped/*/*/*.jpg'
    h5_path = '/home/leifan/h36m-fetch/data/Human3.6/cropped/annot_full.h5'

    h5_file = h5py.File(h5_path, 'r')
    keypoints = h5_file['pose_2d_crop']
    keypoints_3d = h5_file['pose_3d']
    keypoints_3d_diff = h5_file['pose_3d_diff']

    dataset_len = keypoints_3d.shape[0]
    for idx in range(dataset_len):
        ax.clear()
        k_3d = keypoints_3d[idx]
        k_3d_diff = keypoints_3d_diff[idx]

        ax = animate_pose(ax, k_3d, k_3d_diff)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        plt.savefig('temp_full.png')
        input()
    image_size = 128