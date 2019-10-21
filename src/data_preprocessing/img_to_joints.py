import cv2
import numpy as np
import openpifpaf
import pyximport

pyximport.install()

import argparse
import os
import torch
import scipy.misc


class PP:

    def draw_line(self, a, b, img_edges, color):
        a1 = a[0]
        a2 = a[1]
        b1 = b[0]
        b2 = b[1]
        if (a1 != 0 and a2 != 0 and b1 != 0 and b2 != 0):
            cv2.line(img_edges, (int(a1), int(a2)), (int(b1), int(b2)), color, 1)
        return img_edges

    def draw_points(self, points, edges_dir, identity, img_name, output_dir):
        colors = [
            [165, 42, 42],
            [255, 165, 0],
            [255, 215, 0],
            [184, 134, 11],
            [154, 205, 50],
            [0, 100, 0],
            [32, 178, 170],
            [0, 255, 255],
            [30, 144, 255],
            [25, 25, 112],
            [138, 43, 226],
            [139, 0, 139],
            [218, 112, 214],
            [188, 143, 143],
            [253, 245, 230],
            [240, 255, 240],
            [0, 0, 0]
        ]
        img_edges = cv2.imread(os.path.join(edges_dir, identity, img_name))
        img_edges[:, :, :] = 255

        points = points[2:17, 0:2]
        # hends
        color = [0, 0, 100]
        img_edges = self.draw_line(points[1, :], points[3, :], img_edges, color)
        img_edges = self.draw_line(points[3, :], points[5, :], img_edges, color)
        img_edges = self.draw_line(points[5, :], points[7, :], img_edges, color)
        img_edges = self.draw_line(points[2, :], points[4, :], img_edges, color)
        img_edges = self.draw_line(points[4, :], points[6, :], img_edges, color)
        img_edges = self.draw_line(points[6, :], points[8, :], img_edges, color)

        # body
        color = [0, 100, 0]
        img_edges = self.draw_line(points[1, :], points[2, :], img_edges, color)
        img_edges = self.draw_line(points[1, :], points[9, :], img_edges, color)
        img_edges = self.draw_line(points[2, :], points[10, :], img_edges, color)
        img_edges = self.draw_line(points[9, :], points[10, :], img_edges, color)

        # lengs
        color = [100, 0, 0]
        img_edges = self.draw_line(points[9, :], points[11, :], img_edges, color)
        img_edges = self.draw_line(points[11, :], points[13, :], img_edges, color)
        img_edges = self.draw_line(points[10, :], points[12, :], img_edges, color)
        img_edges = self.draw_line(points[12, :], points[14, :], img_edges, color)

        # for i in range(17):
        #     x = points[i, 0]
        #     y = points[i, 1]
        #     if (x != 0 and y != 0):
        #         # img = cv2.imread(os.path.join('/media/martina/Data/School/CTU/thesis/data/mars', '0001', '0001C1T0001F001.jpg'))
        #         cv2.line(img_edges, (int(x) - 1, int(y) - 1), (int(x) + 1, int(y) + 1), colors[i], 2)
        #         cv2.line(img_edges, (int(x) + 1, int(y) - 1), (int(x) - 1, int(y) + 1), colors[i], 2)

        scipy.misc.imsave(
            os.path.join(os.path.join(output_dir, identity, img_name)), img_edges)
        # cv2.line(img, (int(20) - 2, int(20) - 2), (int(20) + 2, int(20) + 2), (0, 20, 200), 2)

    def __init__(self):
        parser = argparse.ArgumentParser()
        openpifpaf.network.nets.cli(parser)
        openpifpaf.decoder.cli(parser, force_complete_pose=False, instance_threshold=0.05)
        args_openpifpaf = parser.parse_args()
        args_openpifpaf.checkpoint = 'resnet101'
        self.device = torch.device('cpu')

        if torch.cuda.is_available():
            self.device = torch.device('cuda')

        model, _ = openpifpaf.network.nets.factory_from_args(args_openpifpaf)
        model = model.to(self.device)
        self.processor = openpifpaf.decoder.factory_from_args(args_openpifpaf, model)

    def process_img(self, img_name: str, identity: str, output_dir: str, keypts_dir: str, edges_dir: str):
        # img_name_short = img_name.split('_')[0] + '.jpg'
        # img = cv2.imread(os.path.join(input_dir, identity, '0001C1T0001F001.jpg'))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # scipy.misc.imsave(os.path.join('/media/martina/Data/School/CTU/thesis/data/mars_edges_with_kpts_selected_20_64x64', img_name_short), img)

        # img_edges = img_edges[:, 63:192, :]
        # processed_image_cpu = openpifpaf.transforms.image_transform(img)
        # processed_image = processed_image_cpu.contiguous().to(self.device, non_blocking=True)
        # fields = self.processor.fields(torch.unsqueeze(processed_image, 0))[0]
        # keypoint_sets, scores = self.processor.keypoint_sets(fields)
        #
        # skeleton_painter = openpifpaf.show.KeypointPainter(show_box=False,
        #                                                    color_connections=True,
        #                                                    markersize=1,
        #                                                    linewidth=6)

        with openpifpaf.show.canvas(show=False) as ax:
            try:
                # arg_max_score = np.argmax(scores)

                keypoint_set = np.load(os.path.join(keypts_dir, identity, img_name + '.npy'))
                # keypoint_set, score = keypoint_sets[arg_max_score, :, :], scores[arg_max_score]

                # # pure joints
                # img = np.zeros(img.shape, dtype=np.uint8)
                # ax.imshow(img)
                # skeleton_painter.keypoints(ax, [keypoint_set], scores=[score])
                # ax.figure.savefig(os.path.join(output_dir,  identity, f'{img_name[:-4]}_{score}.jpg'))

                # joints with edges
                # ax.imshow(img)
                keypoint_set[:, 0] = keypoint_set[:, 0] / 4 + 16
                keypoint_set[:, 1] = keypoint_set[:, 1] / 4
                self.draw_points(keypoint_set, edges_dir, identity, img_name, output_dir)
                # skeleton_painter.keypoints(ax, [keypoint_set], scores=[score])
                # ax.figure.savefig(os.path.join(output_dir, identity, f'{img_name[:-4]}_{score}.jpg'))

                # keypoint_set = [score, keypoint_set]
                # np.save(os.path.join(output_dir, 'key_points', identity, img_name), keypoint_set)
            except:
                print(f'[INFO]..no joints found for identity {identity} image {img_name}')


if __name__ == '__main__':
    # p = PP()
    # p.draw_points()
    output_dir = '/media/martina/Data/School/CTU/thesis/data/mars_key_points_selected_20_64x64'
    input_dir = '/media/martina/Data/School/CTU/thesis/data/mars'
    edges_dir = '/media/martina/Data/School/CTU/thesis/data/mars_edges_selected_20_64x64'
    keypts_dir = '/media/martina/Data/School/CTU/thesis/data/mars_key_points_selected_20'

    # shutil.rmtree(os.path.join(output_dir, 'joints_edges'))
    # shutil.rmtree(os.path.join(output_dir, 'joints'))
    # shutil.rmtree(os.path.join(output_dir, 'key_points'))
    #
    # os.mkdir(os.path.join(output_dir, 'joints_edges'))
    # os.mkdir(os.path.join(output_dir, 'joints'))
    # os.mkdir(os.path.join(output_dir, 'key_points'))

    all_identities = os.listdir(keypts_dir)
    all_identities.sort()
    p = PP()
    os.mkdir(output_dir)
    for identity in all_identities:
        print(identity)
        os.mkdir(os.path.join(output_dir, identity))
        images = os.listdir(os.path.join(edges_dir, identity))
        images.sort()

        for image in images:
            p.process_img(image, identity, output_dir, keypts_dir, edges_dir)
