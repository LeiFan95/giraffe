import io
import os
import cv2
from h5py._hl import base
from h5py._hl.selections2 import read_selections_scalar
# from im2scene.debugtool import var_shape
import lmdb
import time
import glob
import h5py
import torch
import pickle
import string
import random
import logging
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

class LSUNClass(data.Dataset):
    ''' LSUN Class Dataset Class.

    Args:
        dataset_folder (str): path to LSUN dataset
        classes (str): class name
        size (int): image output size
        random_crop (bool): whether to perform random cropping
        use_tanh_range (bool): whether to rescale images to [-1, 1]
    '''

    def __init__(self, dataset_folder,
                 classes='scene_categories/church_outdoor_train_lmdb',
                 size=64, random_crop=False, use_tanh_range=False):
        root = os.path.join(dataset_folder, classes)

        # Define transforms
        if random_crop:
            self.transform = [
                transforms.Resize(size),
                transforms.RandomCrop(size),
            ]
        else:
            self.transform = [
                transforms.Resize((size, size)),
            ]
        self.transform += [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        if use_tanh_range:
            self.transform += [transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(self.transform)

        import time
        t0 = time.time()
        print('Start loading lmdb file ...')
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        cache_file = '_cache_' + ''.join(
            c for c in root if c in string.ascii_letters)
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key in txn.cursor().iternext(
                    keys=True, values=False)]
            pickle.dump(self.keys, open(cache_file, "wb"))
        print('done!')
        t = time.time() - t0
        print('time', t)
        print("Found %d files." % self.length)

    def __getitem__(self, idx):
        try:
            img = None
            env = self.env
            with env.begin(write=False) as txn:
                imgbuf = txn.get(self.keys[idx])

            buf = io.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            data = {
                'image': img
            }
            return data

        except Exception as e:
            print(e)
            idx = np.random.randint(self.length)
            return self.__getitem__(idx)

    def __len__(self):
        return self.length


class ImagesDataset(data.Dataset):
    ''' Default Image Dataset Class.

    Args:
        dataset_folder (str): path to LSUN dataset
        size (int): image output size
        celebA_center_crop (bool): whether to apply the center
            cropping for the celebA and celebA-HQ datasets.
        random_crop (bool): whether to perform random cropping
        use_tanh_range (bool): whether to rescale images to [-1, 1]
    '''

    def __init__(self, dataset_folder,  size=64, celebA_center_crop=False,
                 random_crop=False, use_tanh_range=False):

        self.size = size
        assert(not(celebA_center_crop and random_crop))
        if random_crop:
            self.transform = [
                transforms.Resize(size),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        elif celebA_center_crop:
            if size <= 128:  # celebA
                crop_size = 108
            else:  # celebAHQ
                crop_size = 650
            self.transform = [
                transforms.CenterCrop(crop_size),
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        else:
            self.transform = [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        if use_tanh_range:
            self.transform += [
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(self.transform)

        self.data_type = os.path.basename(dataset_folder).split(".")[-1]
        assert(self.data_type in ["jpg", "png", "npy"])

        import time
        t0 = time.time()
        print('Start loading file addresses ...')
        images = glob.glob(dataset_folder)
        random.shuffle(images)
        t = time.time() - t0
        print('done! time:', t)
        print("Number of images found: %d" % len(images))

        self.images = images
        self.length = len(images)

    def __getitem__(self, idx):
        try:
            buf = self.images[idx]
            if self.data_type == 'npy':
                img = np.load(buf)[0].transpose(1, 2, 0)
                img = Image.fromarray(img).convert("RGB")
            else:
                img = Image.open(buf).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)
            data = {
                'image': img
            }
            return data
        except Exception as e:
            print(e)
            print("Warning: Error occurred when loading file %s" % buf)
            return self.__getitem__(np.random.randint(self.length))

    def __len__(self):
        return self.length



class H36MDataset(data.Dataset):
    def __init__(self, idx_txt, dataset_folder, h5_path, image_size=128,
            use_tanh_range=False, num_parts=41, relations=None, mode='train'):
        """Initialize and preprocess the CelebA dataset."""
        self.idx_txt = idx_txt
        self.dataset_folder = dataset_folder
        self.h5_path = h5_path
        self.image_size = image_size
        self.num_parts = num_parts
        self.relations = relations
        self.mode = mode
        
        t0 = time.time()
        print('Start loading file addresses ...')

        with open(self.idx_txt, 'r') as idx_file:
            image_names = idx_file.read().splitlines()
        self.image_dict = {}
        for idx, image_name in enumerate(image_names):
            self.image_dict[image_name] = idx

        images = glob.glob(dataset_folder)
        random.shuffle(images)

        h5_file = h5py.File(h5_path, 'r')
        self.keypoints = h5_file['pose_2d_crop']
        self.keypoints_3d = h5_file['pose_3d']
        self.keypoints_3d_diff = h5_file['pose_3d_diff']

        t = time.time() - t0
        print('done! time:', t)
        print("Number of images found: %d" % len(images))

        self.generate_heatmap = GenerateHeatmap(self.image_size, 
                                    self.num_parts, self.relations, with_skeleton=True)

        self.dataset = images
        self.num_images = len(self.dataset)

        self.transform = []
        self.transform.append(transforms.ToTensor())
        self.transform.append(transforms.Resize([self.image_size, self.image_size]))
        if use_tanh_range:
            self.transform += [
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(self.transform)
        
        self.transform_skeleton = [
                transforms.ToTensor(),
                transforms.Resize([16, 16])
        ]
        self.transform_skeleton = transforms.Compose(self.transform_skeleton)

    def __getitem__(self, idx):
        """Return one image and its corresponding attribute label."""
        dataset = self.dataset
        filename = dataset[idx]
        image = Image.open(filename).convert('RGB')
        height, width = image.height, image.width
        h_scale, w_scale = self.image_size/height, self.image_size/width
        h_center, w_center = self.image_size/2., self.image_size/2.

        idx = self.image_dict[filename]
        keypoint = np.array(self.keypoints[idx])
        keypoint_3d = np.array(self.keypoints_3d[idx])
        keypoint_3d_diff = np.array(self.keypoints_3d_diff[idx])

        image = self.transform(image)

        # transform keypoint
        keypoint[:, 0], keypoint[:, 1] = (keypoint[:, 0]*w_scale).astype(int), (keypoint[:, 1]*h_scale).astype(int)
        heatmaps = self.generate_heatmap(keypoint)

        # normalize keypoint to [-1., 1.]
        # keypoint[:, 0], keypoint[:, 1] = (keypoint[:, 0] - w_center)/w_center, (keypoint[:, 1] - h_center)/h_center
        keypoint = torch.from_numpy(keypoint).type(torch.FloatTensor)
        keypoint_3d = torch.from_numpy(keypoint_3d).type(torch.FloatTensor)
        keypoint_3d_diff = torch.from_numpy(keypoint_3d_diff).type(torch.FloatTensor)

        skeleton = self.transform_skeleton(heatmaps[-1]).permute(0, 2, 1)
        heatmaps = heatmaps[:-1]
        data = {
            'image': image,
            'keypoint': keypoint,
            'keypoint_3d': keypoint_3d,
            'keypoint_3d_diff': keypoint_3d_diff,
            'heatmaps': heatmaps,
            'skeleton': skeleton,
        }
        return data

    def __len__(self):
        """Return the number of images."""
        return self.num_images
    
    def get_sampled_pose(self, sample_size=16):
        assert sample_size <= self.num_images
        sampled_poses = np.zeros((sample_size, self.num_parts, 3))
        sampled_poses_diff = np.zeros((sample_size, self.num_parts, 3))
        sampled_skeletons = np.zeros((sample_size, 16, 16))
        for idx in range(sample_size):
            keypoint = np.array(self.__getitem__(idx)['keypoint_3d'])
            keypoint_diff = np.array(self.__getitem__(idx)['keypoint_3d_diff'])
            skeleton = np.array(self.__getitem__(idx)['skeleton'])
            sampled_poses[idx] = keypoint
            sampled_poses_diff[idx] = keypoint_diff
            sampled_skeletons[idx, :, :] = skeleton
        sampled_poses = torch.tensor(sampled_poses).type('torch.FloatTensor')
        sampled_poses_diff = torch.tensor(sampled_poses_diff).type('torch.FloatTensor')
        sampled_skeletons = torch.tensor(sampled_skeletons).type('torch.FloatTensor')
        return sampled_poses, sampled_poses_diff, sampled_skeletons

    def drawKeypoints(self, image, keypoints):
        norm = np.zeros((image.shape[0], image.shape[1]))
        image = cv2.normalize(image, norm, 0, 255, cv2.NORM_MINMAX)
        for idx, k in enumerate(keypoints):
            cv2.circle(image, (int(k[0]), int(k[1])), 2, (255, 0, 0), thickness=5)
            cv2.putText(image, str(idx), (int(k[0]), int(k[1])), cv2.FONT_HERSHEY_SIMPLEX, \
                fontScale=0.3, color=(255, 255, 255), thickness=1)
        return image


class EvalDataset(data.Dataset):
    def __init__(self, h5_path_eval):
        """Initialize and preprocess the CelebA dataset."""
        self.h5_path_eval = h5_path_eval
        
        h5_file_eval = h5py.File(h5_path_eval, 'r')
        self.keypoints = h5_file_eval['pose_2d_crop']
        self.keypoints_3d = h5_file_eval['pose_3d']
        self.keypoints_3d_diff = h5_file_eval['pose_3d_diff']
        self.num_samples = self.keypoints.shape[0]

    def __getitem__(self, idx):
        keypoint = np.array(self.keypoints[idx])
        keypoint_3d = np.array(self.keypoints_3d[idx])
        keypoint_3d_diff = np.array(self.keypoints_3d_diff[idx])

        keypoint = torch.from_numpy(keypoint).type(torch.FloatTensor)
        keypoint_3d = torch.from_numpy(keypoint_3d).type(torch.FloatTensor)
        keypoint_3d_diff = torch.from_numpy(keypoint_3d_diff).type(torch.FloatTensor)
        data = {
            'keypoint': keypoint,
            'keypoint_3d': keypoint_3d,
            'keypoint_3d_diff': keypoint_3d_diff,
        }
        return data

    def __len__(self):
        return self.num_samples

    def get_movement_seq(self, sample_size=16):
        move_keypoint_3d = np.zeros((sample_size, self.num_parts, 3))
        move_keypoint_3d_diff = np.zeros((sample_size, self.num_parts, 3))
        start_idx, end_idx = random.randint(0, self.num_samples), random.randint(0, self.num_samples)
        start_keypoint_3d, end_keypoint_3d = self.keypoints_3d[start_idx], self.keypoints_3d[end_idx]
        start_keypoint_3d_diff, end_keypoint_3d_diff = self.keypoint_3d_diff[start_idx], self.keypoint_3d_diff[end_idx]

        movement_3d = (end_keypoint_3d - start_keypoint_3d)/(sample_size - 1)
        movement_3d_diff = (end_keypoint_3d_diff - start_keypoint_3d_diff)/(sample_size - 1)
        for idx in range(sample_size):
            move_keypoint_3d[idx] = start_keypoint_3d + movement_3d*idx
            move_keypoint_3d_diff[idx] = start_keypoint_3d_diff + movement_3d_diff*idx
        return move_keypoint_3d, move_keypoint_3d_diff

class GenerateHeatmap():
    def __init__(self, output_res, num_parts, relations, with_skeleton=False):
        self.output_res = output_res
        self.num_parts = num_parts
        self.relations = relations
        self.with_skeleton = with_skeleton
        sigma = self.output_res/24
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    
    def __call__(self, keypoints):
        if self.with_skeleton:
            hms = np.zeros((self.num_parts + 1, self.output_res, self.output_res), dtype = np.float32)
        else:
            hms = np.zeros((self.num_parts, self.output_res, self.output_res), dtype = np.float32)
        sigma = self.sigma
        for idx, pt in enumerate(keypoints):
            if pt[0] > 0 and pt[1] > 0 and pt[0] < self.output_res - 1 and pt[1] < self.output_res - 1:
                x, y = int(pt[0]), int(pt[1])
                if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                    continue
                ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        if self.with_skeleton:
            map = self.skeleton(keypoints)
            for idx, pt in enumerate(np.transpose(np.nonzero(map))):
                x, y = int(pt[1]), int(pt[0])
                if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                    continue
                ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                map[aa:bb, cc:dd] = np.maximum(map[aa:bb, cc:dd], self.g[a:b, c:d])
            hms[-1, :, :] = map[:, :]
        return hms

    def skeleton(self, keypoints):
        map = np.zeros((self.output_res, self.output_res), dtype = np.float32)
        for idx, rel in enumerate(self.relations):
            pt_1, pt_2 = keypoints[rel[0]], keypoints[rel[1]]
            cv2.line(map, (int(pt_1[0]), int(pt_1[1])), (int(pt_2[0]), int(pt_2[1])), color=255, thickness=1)
        return map

def draw_keypoints(image, keypoints, relations_h36m, thickness=1, normalize=False):
    if normalize:
        norm = np.zeros((image.shape[0], image.shape[1], 3))
        image = cv2.normalize(image, norm, 0, 255, cv2.NORM_MINMAX)
    
    # print('image', image.shape)
    for idx, rel in enumerate(relations_h36m):
        pt_1, pt_2 = keypoints[rel[0]], keypoints[rel[1]]
        # print(pt_1, pt_2)
        # input()
        cv2.line(image, (int(pt_1[0]), int(pt_1[1])), (int(pt_2[0]), int(pt_2[1])), color=(0, 0, 255), thickness=1)
    for idx, k in enumerate(keypoints):
        cv2.circle(image, (int(k[0]), int(k[1])), 1, (255, 0, 0), thickness=thickness)
        cv2.putText(image, str(idx), (int(k[0]), int(k[1])), cv2.FONT_HERSHEY_SIMPLEX, \
            fontScale=0.3, color=(255, 255, 255), thickness=thickness)
    return image

def render_3D_pose(ax, coordinates, relations_h36m):
    relations_h36m = [[0, 1], [0, 4], [1, 2], [2, 3], [4, 5], [5, 6], 
        [0, 7], [7, 8], [8, 11], [8, 14], [14, 15], 
        [15, 16], [11, 12], [12, 13], [10, 9], [9, 8]]
    xs, ys, zs = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    for rel in relations_h36m:
        idx0, idx1 = rel[0], rel[1]
        ax.plot([xs[idx0],xs[idx1]], [ys[idx0],ys[idx1]], [zs[idx0],zs[idx1]], linewidth=1, label=r'$z=y=x$')
    ax.scatter(xs, ys, zs)
    return ax

if __name__ == '__main__':
    import torch
    idx_txt = '/home/leifan/h36m-fetch/data/Human3.6/cropped/names.txt'
    dataset_folder = '/data/leifan/dataset/Human3.6/cropped/*/*/*.jpg'
    h5_path = '/home/leifan/h36m-fetch/data/Human3.6/cropped/annot_new.h5'
    image_size = 128
    use_tanh_range = False
    relations_h36m = [[0, 1], [0, 4], [1, 2], [2, 3], [4, 5], [5, 6], 
        [0, 7], [7, 8], [8, 11], [8, 14], [14, 15], 
        [15, 16], [11, 12], [12, 13], [10, 9], [9, 8]]
    mode = 'train'
    dataset = H36MDataset(idx_txt, dataset_folder, h5_path, \
        image_size=image_size, use_tanh_range=use_tanh_range, relations=relations_h36m, mode=mode)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=1, shuffle=True,
        pin_memory=True, drop_last=True,
    )
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    major_locator = MultipleLocator(0.25)
    ax.xaxis.set_major_locator(major_locator)
    ax.yaxis.set_major_locator(major_locator)
    ax.zaxis.set_major_locator(major_locator)
    for batch in train_loader:
        ax.clear()
        x_real = batch.get('image').squeeze(0).permute(1, 2, 0).numpy()
        keypoint = batch.get('keypoint').squeeze(0).reshape(17, 2).numpy()
        keypoint_3d = batch.get('keypoint_3d').squeeze(0).reshape(17, 3).numpy()
        skeleton = batch.get('skeleton').squeeze(0).squeeze(0).numpy()
        print('skeleton', skeleton.shape)
        cv2.imwrite('skeleton.png', (skeleton*255).astype(int))
        # input()
        
        result = draw_keypoints(x_real, keypoint, relations_h36m, normalize=True)
        cv2.imwrite('temp.png', result)
        ax = render_3D_pose(ax, keypoint_3d, relations_h36m)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        plt.savefig('temp3d.png')
        input()