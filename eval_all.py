import torch
import os
import argparse
from tqdm import tqdm
import time
from im2scene import config
from im2scene.checkpoints import CheckpointIO
import numpy as np
from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)
from math import ceil
from torchvision.utils import save_image, make_grid


parser = argparse.ArgumentParser(
    description='Evaluate a GIRAFFE model from different aspects.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--cuda_id', '-id', type=str, default='0', help='Available GPU index.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
n_workers = cfg['training']['n_workers']
out_dict_file = os.path.join(out_dir, 'fid_evaluation.npz')
out_img_file = os.path.join(out_dir, 'fid_images.npy')
out_vis_file = os.path.join(out_dir, 'fid_images.jpg')

# Model
model = config.get_model(cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Generate
model.eval()
generator = model.generator

fid_file = cfg['data']['fid_file']
assert(fid_file is not None)
fid_dict = np.load(cfg['data']['fid_file'])

n_images = cfg['test']['n_images']
move_seq_size = cfg['test']['move_seq_size']
batch_size = cfg['training']['batch_size']
n_iter = ceil(n_images/batch_size)

eval_dataset = config.get_dataset(cfg)
eval_loader = torch.utils.data.DataLoader(
    eval_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True,
    pin_memory=True, drop_last=True,
)

# generate movements
vis_dict = model.generator.get_vis_dict(move_seq_size)
move_keypoint_3d, move_keypoint_3d_diff = eval_dataset.get_movement_seq(sample_size=move_seq_size)
t0 = time.time()

with torch.no_grad():
    move_fake, alpha_map = generator(move_keypoint_3d, move_keypoint_3d_diff, **vis_dict, 
                                        mode='val', return_alpha_map=True)
    move_fake = move_fake.cpu()
    alpha_map = alpha_map.cpu()

img_fake = torch.cat(img_fake, dim=0)[:n_images]
img_fake.clamp_(0., 1.)
n_images = img_fake.shape[0]
