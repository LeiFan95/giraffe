import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

def get_can_pose(batch_size=1, device='cuda'):
    can_pose = np.array([[0., -0.18, 0.], 
                    [0.17, -0.18, 0.],
                    [0.17, 0.37, 0.],
                    [0.17, 0.98, 0.0],
                    [-0.17, -0.18, 0.],
                    [-0.17, 0.37, 0.],
                    [-0.17, 0.95, 0.0],
                    [0., -0.42, 0.],
                    [0., -0.74, 0.],
                    [0., -0.89, 0.08],
                    [0., -0.98, 0.],
                    [-0.17, -0.6, 0.],
                    [-0.22, -0.3, 0.],
                    [-0.27, -0.1, 0.],
                    [0.17, -0.6, 0.],
                    [0.22, -0.3, 0.],
                    [0.27, -0.1, 0.]])
    can_pose_new = can_pose.copy()
    can_pose_new[:, 0] = can_pose[:, 0]
    can_pose_new[:, 1] = can_pose[:, 2]
    can_pose_new[:, 2] = -can_pose[:, 1]
    
    can_pose_new = torch.from_numpy(can_pose_new).type(torch.FloatTensor).cuda()
    can_pose_new = can_pose_new.unsqueeze(0)
    can_pose_new = can_pose_new.repeat(batch_size, 1, 1)
    return can_pose_new

def var_print(name, var):
    print('shape:')
    var_shape(name, var)
    print(name, 'value:')
    print(var)
    input()

def var_shape(name, var):
    if torch.is_tensor(var):
        print('tensor', name, list(var.shape))
    elif isinstance(var, np.ndarray):
        print('np.ndarray', name, var.shape)
    elif isinstance(var, list):
        print('list', name, len(var))
    else:
        print('unknown', name, var)
    input()

def draw_keypoints(image, keypoints, thickness=1, normalize=False):
    relations_h36m = [[0, 4], [0, 7], [0, 1], [1, 2], 
        [2, 3], [4, 5], [5, 6], [7, 8], [8, 9], [9, 10], 
        [8, 11], [8, 14], [11, 12], [12, 13], [14, 15], [15, 16]]
    if normalize:
        norm = np.zeros((image.shape[0], image.shape[1], 3))
        image = cv2.normalize(image, norm, 0, 255, cv2.NORM_MINMAX)
    
    for idx, rel in enumerate(relations_h36m):
        pt_1, pt_2 = keypoints[rel[0]], keypoints[rel[1]]
        cv2.line(image, (int(pt_1[0]), int(pt_1[1])), (int(pt_2[0]), int(pt_2[1])), color=(0, 0, 255), thickness=1)
    for idx, k in enumerate(keypoints):
        cv2.circle(image, (int(k[0]), int(k[1])), 1, (255, 0, 0), thickness=thickness)
        cv2.putText(image, str(idx), (int(k[0]), int(k[1])), cv2.FONT_HERSHEY_SIMPLEX, \
            fontScale=0.3, color=(255, 255, 255), thickness=thickness)
    return image

def np_gallery(array, ncols=4):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

# 3d pose visulizer
def save_3d_pose_image(vis_dir, name, type, poses_3d, nrow=4, elev=None, azim=None):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    from mpl_toolkits.mplot3d import Axes3D
    relations = [[0, 1], [0, 4], [1, 2], [2, 3], [4, 5], [5, 6], 
            [0, 7], [7, 8], [8, 11], [8, 14], [14, 15], 
            [15, 16], [11, 12], [12, 13], [10, 9], [9, 8]]

    fig = plt.figure()
    gcf = plt.gcf()
    gcf.set_size_inches(18.5, 18.5)
    ncol = poses_3d.shape[0]//nrow
    for idx in range(poses_3d.shape[0]):
        # print(nrow*100 + ncol*10 + idx + 1)
        ax = fig.add_subplot(nrow, ncol, idx + 1, projection='3d')
        if elev != None and azim != None:
            ax.view_init(elev=elev, azim=azim)
        major_locator = MultipleLocator(0.25)
        ax.xaxis.set_major_locator(major_locator)
        ax.yaxis.set_major_locator(major_locator)
        ax.zaxis.set_major_locator(major_locator)
        pose_3d = poses_3d[idx].reshape(41, 3)
        xs, ys, zs = pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2]
        for rel in relations:
            idx0, idx1 = rel[0], rel[1]
            ax.plot([xs[idx0],xs[idx1]], [ys[idx0],ys[idx1]], [zs[idx0],zs[idx1]], 
                        linewidth=1, label=r'$z=y=x$')
        ax.scatter(xs, ys, zs)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
    pose_3d_path = os.path.join(vis_dir, type)
    if not os.path.exists(pose_3d_path):
        os.makedirs(pose_3d_path)
    plt.savefig(os.path.join(pose_3d_path, name))

# image: [bc_sz * c * img_sz * img_sz]
def save_grid_image(vis_dir, name, type, image, nrow=4, save=True):
    if type == 'alpha':
        image = F.interpolate(image, scale_factor=4)
    image_grid = make_grid(image.clamp_(0., 1.), nrow=nrow)
    save_path = os.path.join(vis_dir, type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save:
        save_image(image_grid, os.path.join(save_path, name))
    return image_grid