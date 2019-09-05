import cv2
import numpy as np
import openpifpaf
import pyximport

pyximport.install()

import argparse
import os
import torch
import timeit
import scipy.misc


class PP:

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

    def process_img(self, img_name: str, identity: str, output_dir: str, input_dir: str, edges_dir: str):
        img_name_short = img_name.split('_')[0] + '.jpg'
        img = cv2.imread(os.path.join(input_dir, identity, img_name_short))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        scipy.misc.imsave(os.path.join('/media/martina/Data/School/CTU/thesis/data/mars_edges_with_kpts_selected_20_64x64', img_name_short), img)

        img_edges = cv2.imread(os.path.join(edges_dir, identity, img_name))
        # img_edges = img_edges[:, 63:192, :]
        processed_image_cpu = openpifpaf.transforms.image_transform(img)
        processed_image = processed_image_cpu.contiguous().to(self.device, non_blocking=True)
        fields = self.processor.fields(torch.unsqueeze(processed_image, 0))[0]
        keypoint_sets, scores = self.processor.keypoint_sets(fields)

        skeleton_painter = openpifpaf.show.KeypointPainter(show_box=False,
                                                           color_connections=True,
                                                           markersize=1,
                                                           linewidth=6)

        with openpifpaf.show.canvas(show=False) as ax:
            try:
                arg_max_score = np.argmax(scores)
                keypoint_set, score = keypoint_sets[arg_max_score, :, :], scores[arg_max_score]

                # # pure joints
                # img = np.zeros(img.shape, dtype=np.uint8)
                # ax.imshow(img)
                # skeleton_painter.keypoints(ax, [keypoint_set], scores=[score])
                # ax.figure.savefig(os.path.join(output_dir,  identity, f'{img_name[:-4]}_{score}.jpg'))

                # joints with edges
                ax.imshow(img_edges)
                keypoint_set[:, 0] = keypoint_set[:, 0] / 4 + 16
                keypoint_set[:, 1] = keypoint_set[:, 1] / 4
                skeleton_painter.keypoints(ax, [keypoint_set], scores=[score])
                ax.figure.savefig(os.path.join(output_dir, identity, f'{img_name[:-4]}_{score}.jpg'))

                # keypoint_set = [score, keypoint_set]
                # np.save(os.path.join(output_dir, 'key_points', identity, img_name), keypoint_set)
            except:
                print(f'[INFO]..no joints found for identity {identity} image {img_name}')


if __name__ == '__main__':
    output_dir = '/media/martina/Data/School/CTU/thesis/data/mars_edges_with_kpts_selected_20_64x64'
    input_dir = '/media/martina/Data/School/CTU/thesis/data/mars'
    edges_dir = '/media/martina/Data/School/CTU/thesis/data/mars_edges_selected_20_64x64'

    # shutil.rmtree(os.path.join(output_dir, 'joints_edges'))
    # shutil.rmtree(os.path.join(output_dir, 'joints'))
    # shutil.rmtree(os.path.join(output_dir, 'key_points'))
    #
    # os.mkdir(os.path.join(output_dir, 'joints_edges'))
    # os.mkdir(os.path.join(output_dir, 'joints'))
    # os.mkdir(os.path.join(output_dir, 'key_points'))

    all_identities = os.listdir(input_dir)
    processed_identities = os.listdir(output_dir)
    # [all_identities.remove(identity) for identity in processed_identities]
    all_identities.sort()
    p = PP()
    for identity in all_identities:
        print(identity)
        os.mkdir(os.path.join(output_dir, identity))
        images = os.listdir(os.path.join(edges_dir, identity))
        images.sort()

        for image in images:
            p.process_img(image, identity, output_dir, input_dir, edges_dir)
        print(f'[INFO]..{identity} loaded')
