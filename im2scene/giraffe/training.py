from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)
from im2scene.training import (
    toggle_grad, compute_grad2, compute_bce, toggle_grad_t, update_average)
from im2scene.debugtool import (
    var_shape, draw_keypoints, np_gallery, save_3d_pose_image, save_grid_image)
import os
import cv2
import torch
from torchvision.utils import save_image
from im2scene.training import BaseTrainer
from tqdm import tqdm
import logging
logger_py = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    ''' Trainer object for GIRAFFE.

    Args:
        model (nn.Module): GIRAFFE model
        optimizer (optimizer): generator optimizer object
        optimizer_d (optimizer): discriminator optimizer object
        device (device): pytorch device
        vis_dir (str): visualization directory
        multi_gpu (bool): whether to use multiple GPUs for training
        fid_dict (dict): dicionary with GT statistics for FID
        n_eval_iterations (int): number of eval iterations
        overwrite_visualization (bool): whether to overwrite
            the visualization files
    '''

    def __init__(self, model, optimizer, optimizer_d,
                 device=None, vis_dir=None, visualize_every=1000,
                 multi_gpu=False, fid_dict={},
                 n_eval_iterations=10,
                 overwrite_visualization=True, **kwargs):
        assert multi_gpu == False, 'current not support multi-gpu'
        self.model = model
        self.optimizer = optimizer
        self.optimizer_d = optimizer_d
        self.device = device
        self.vis_dir = vis_dir
        self.multi_gpu = multi_gpu
        self.visualize_every = visualize_every

        self.overwrite_visualization = overwrite_visualization
        self.fid_dict = fid_dict
        self.n_eval_iterations = n_eval_iterations

        self.vis_dict = model.generator.get_vis_dict(16)

        self.generator = self.model.generator
        self.discriminator = self.model.discriminator
        self.generator_test = self.model.generator_test

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, it=None):
        loss_g = self.train_step_generator(data, it)
        # loss_g_static = self.train_step_generator(data, it, train_static=True)
        loss_d, reg_d, fake_d, real_d = self.train_step_discriminator(data, it)

        return {
            'generator': loss_g,
            # 'generator_static': loss_g_static,
            'discriminator': loss_d,
            'regularizer': reg_d,
        }

    def eval_step(self, val_loader):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()

        x_fake = []
        it = 0
        n_iter = self.n_eval_iterations
        for data in val_loader:
            if it > n_iter:
                break
            it += 1
            keypoint_3d = data.get('keypoint_3d').to(self.device)
            keypoint_3d_diff = data.get('keypoint_3d_diff').to(self.device)
            skeleton = data.get('skeleton').to(self.device)
            with torch.no_grad():
                x_fake.append(gen(keypoint_3d, keypoint_3d_diff, skeleton=skeleton).cpu()[:, :3])

        x_fake = torch.cat(x_fake, dim=0)
        x_fake.clamp_(0., 1.)
        mu, sigma = calculate_activation_statistics(x_fake)
        fid_score = calculate_frechet_distance(
            mu, sigma, self.fid_dict['m'], self.fid_dict['s'], eps=1e-4)
        eval_dict = {
            'fid_score': fid_score
        }
        return eval_dict

    def train_step_generator(self, data, it=None, z=None, train_static=False):
        self.generator = toggle_grad(self.generator, True, mode='train')
        self.discriminator = toggle_grad(self.discriminator, False, mode='train')
        self.optimizer.zero_grad()

        keypoint_3d = data.get('keypoint_3d').to(self.device)
        keypoint_3d_diff = data.get('keypoint_3d_diff').to(self.device)
        skeleton = data.get('skeleton').to(self.device)

        if self.multi_gpu:
            latents = self.generator.module.get_vis_dict()
            x_fake = self.generator(keypoint_3d, keypoint_3d_diff, **latents, skeleton=skeleton, train_static=train_static)
        else:
            x_fake = self.generator(keypoint_3d, keypoint_3d_diff, skeleton=skeleton, train_static=train_static)

        d_fake = self.discriminator(x_fake)
        gloss = compute_bce(d_fake, 1)

        gloss.backward()
        self.optimizer.step()

        if self.generator_test is not None:
            update_average(self.generator_test, self.generator, beta=0.999)
        return gloss.item()

    def train_step_discriminator(self, data, it=None, z=None):
        self.generator = toggle_grad(self.generator, False, mode='train')
        self.discriminator = toggle_grad(self.discriminator, True, mode='train')
        self.optimizer_d.zero_grad()

        x_real = data.get('image').to(self.device)
        keypoint = data.get('keypoint').to(self.device)
        keypoint_3d = data.get('keypoint_3d').to(self.device)
        keypoint_3d_diff = data.get('keypoint_3d_diff').to(self.device)
        loss_d_full = 0.

        x_real.requires_grad_()
        d_real = self.discriminator(x_real)

        d_loss_real = compute_bce(d_real, 1)
        loss_d_full += d_loss_real

        reg = 10. * compute_grad2(d_real, x_real).mean()
        loss_d_full += reg

        with torch.no_grad():
            if self.multi_gpu:
                latents = self.generator.module.get_vis_dict()
                x_fake = self.generator(keypoint_3d, keypoint_3d_diff, **latents)
            else:
                x_fake = self.generator(keypoint_3d, keypoint_3d_diff)

        x_fake.requires_grad_()
        d_fake = self.discriminator(x_fake)

        d_loss_fake = compute_bce(d_fake, 0)
        loss_d_full += d_loss_fake

        loss_d_full.backward()
        self.optimizer_d.step()

        d_loss = (d_loss_fake + d_loss_real)

        return (d_loss.item(), reg.item(), d_loss_fake.item(), d_loss_real.item())

    # def train_step_pose_estimator(self, data, it=None):
    #     self.generator = toggle_grad(self.generator, False, 'train')
    #     self.discriminator = toggle_grad(self.discriminator, False, 'train')
    #     self.pose_estimator = toggle_grad(self.pose_estimator, True, 'train')

    #     self.optimizer_pose.zero_grad()
    #     x_real = data.get('image').to(self.device)
    #     heatmaps = data.get('heatmaps').to(self.device)

    #     x_real.requires_grad_()
    #     pose_real = self.pose_estimator(x_real)

    #     # visualization of training data
    #     if it % self.visualize_every == 0:
    #         base_name = '%010d.png' % it
    #         hm_img = torch.sum(heatmaps, dim=1, keepdim=True).repeat(1, 3, 1, 1)
    #         hm_img = torch.max(x_real, hm_img)
    #         self.save_grid_image(base_name, 'pose_est_inp', hm_img)
    #         self.save_pose_image(base_name, 'pose_est_out', pose_real)

    #     # TODO the constant multiplier of each value
    #     loss_pose = torch.sum(torch.mean(self.pose_estimator.calc_loss(pose_real, heatmaps), dim=-1))
    #     loss_pose.backward()
    #     self.optimizer_pose.step()
    #     return loss_pose.item()

    # direct pose estimator output
    def save_pose_image(self, name, type, pose, nrow=4):
        pose = self.pose_estimator.visualize(pose)
        pose = np_gallery(pose)
        pose_path = os.path.join(self.vis_dir, type)
        if not os.path.exists(pose_path):
            os.makedirs(pose_path)
        cv2.imwrite(os.path.join(pose_path, name), pose)
        return pose

    def visualize(self, sampled_tuple=None, it=0):
        ''' Visualized the data.

        Args:
            it (int): training iteration
        '''
        assert sampled_tuple != None
        sampled_pose, sampled_pose_diff, sampled_skeleton = sampled_tuple
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()

        with torch.no_grad():
            sampled_pose = sampled_pose.to(self.device)
            sampled_pose_diff = sampled_pose_diff.to(self.device)
            sampled_skeleton = sampled_skeleton.to(self.device).unsqueeze(1)
            image_fake, alpha_map = self.generator(sampled_pose, sampled_pose_diff, **self.vis_dict, mode='val', 
                                                    return_alpha_map=True, skeleton=sampled_skeleton,
                                                    only_render_background=False)
            image_fake_bg = self.generator(sampled_pose, sampled_pose_diff, **self.vis_dict, mode='val', 
                                                    return_alpha_map=False, skeleton=sampled_skeleton,
                                                    only_render_background=True)
            image_fake_fg = self.generator(sampled_pose, sampled_pose_diff, **self.vis_dict, mode='val', 
                                                    return_alpha_map=False, skeleton=sampled_skeleton,
                                                    only_render_background=False, not_render_background=True)
                                                    # train_static=True, only_render_background=True)
            image_fake = image_fake.cpu()
            alpha_map = alpha_map.cpu()
            image_fake_bg = image_fake_bg.cpu()
            image_fake_fg = image_fake_fg.cpu()

        base_name = '%010d.png' % it
        # pose_fake_grid = self.save_pose_image(base_name, 'pose_syn', pose_fake)
        image_fake_grid = save_grid_image(self.vis_dir, base_name, 'syn', image_fake, save=False)
        alpha_map_grid = save_grid_image(self.vis_dir, base_name, 'alpha', alpha_map, save=False)
        image_fake_grid_bg = save_grid_image(self.vis_dir, base_name, 'syn_bg', image_fake_bg, save=False)
        image_fake_grid_fg = save_grid_image(self.vis_dir, base_name, 'syn_fg', image_fake_fg, save=False)
        
        all_grid = torch.cat((alpha_map_grid, image_fake_grid_bg, 
                                image_fake_grid_fg, image_fake_grid), dim=2)
        save_all_grid_path = os.path.join(self.vis_dir, 'all')
        if not os.path.exists(save_all_grid_path):
            os.makedirs(save_all_grid_path)
        save_image(all_grid, os.path.join(save_all_grid_path, base_name))
        return image_fake_grid