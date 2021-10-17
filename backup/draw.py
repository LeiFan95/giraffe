
#####################################
                temp = torch.all(delta_p_i[0] != 0, dim=-1)
                import matplotlib.pyplot as plt
                from matplotlib.pyplot import MultipleLocator
                from mpl_toolkits.mplot3d import Axes3D
                
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.clear()
                major_locator = MultipleLocator(0.25)
                
                d_p = p_i[0, temp].cpu().numpy()
                d_p_new = p_i_new[0, temp].cpu().numpy()
                ax.xaxis.set_major_locator(major_locator)
                ax.yaxis.set_major_locator(major_locator)
                ax.zaxis.set_major_locator(major_locator)

                for idx in range(d_p.shape[0]):
                    ax.plot([d_p[idx, 0], d_p_new[idx, 0]], [d_p[idx, 1], d_p_new[idx, 1]], [d_p[idx, 2], d_p_new[idx, 2]], linewidth=1, label=r'$z=y=x$')
                ax.scatter(d_p[:, 0], d_p[:, 1], d_p[:, 2], c='r')
                ax.scatter(d_p_new[:, 0], d_p_new[:, 1], d_p_new[:, 2], c='b')

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
                plt.savefig('sampled_points.png')
                input()
######################################

#################################################
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import MultipleLocator
        import numpy as np
        relations = [[0, 1], [0, 4], [1, 2], [2, 3], [4, 5], [5, 6], 
                [0, 7], [7, 8], [8, 11], [8, 14], [14, 15], 
                [15, 16], [11, 12], [12, 13], [10, 9], [9, 8]]

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
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        major_locator = MultipleLocator(0.25)
        ax = animate_pose(ax, keypoint_3d.cpu().numpy()[0], keypoint_3d_diff.cpu().numpy()[0])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        plt.savefig('skeleton.png')
        print('skeleton saved')
        input()
        
#################################################