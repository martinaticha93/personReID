import cv2
import numpy as np
import openpifpaf
import pyximport

pyximport.install()

import argparse
import os
import torch
import timeit


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
        img = cv2.imread(os.path.join(input_dir, identity, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            img_edges = cv2.imread(os.path.join(edges_dir, identity, img_name))
            img_edges = img_edges[:, 63:192, :]
            processed_image_cpu = openpifpaf.transforms.image_transform(img)
            processed_image = processed_image_cpu.contiguous().to(self.device, non_blocking=True)
            fields = self.processor.fields(torch.unsqueeze(processed_image, 0))[0]
            # keypoint_sets, scores = self.processor.keypoint_sets(fields)

            skeleton_painter = openpifpaf.show.KeypointPainter(show_box=False,
                                                               color_connections=True,
                                                               markersize=1,
                                                               linewidth=6)

            with openpifpaf.show.canvas(show=False) as ax:
                try:
                    # arg_max_score = np.argmax(scores)
                    # keypoint_set, score = keypoint_sets[arg_max_score, :, :], scores[arg_max_score]

                    # pure joints
                    img = np.zeros(img.shape, dtype=np.uint8)
                    ax.imshow(img)

                    # joints with edges
                    ax.imshow(img_edges)
                    # skeleton_painter.keypoints(ax, [keypoint_set], scores=[score])
                    # ax.figure.savefig(
                    #     os.path.join(output_dir, 'joints_edges_original', identity, f'{img_name[:-4]}_{score}.jpg'))

                    # keypoint_set = [score, keypoint_set]

                    # np.save(os.path.join(output_dir, 'key_points', identity, img_name), keypoint_set)
                    [score, keypoint_set] = np.load(os.path.join(output_dir, 'key_points', identity, img_name + '.npy'))
                    skeleton_painter.keypoints(ax, [keypoint_set], scores=[score])
                    ax.figure.savefig(os.path.join(output_dir, 'joints_edges_original', identity, f'{img_name[:-4]}_{score}.jpg'))
                except:
                    print(f'[INFO]..no joints found for identity {identity} image {img_name}')
        except:
            print(f'[INFO]..image {img_name} for identity {identity} could not have been tranformed')


if __name__ == '__main__':
    output_dir = '/media/martina/Data/School/CTU/thesis/mars_joints'
    input_dir = '/media/martina/Data/School/CTU/thesis/mars'
    edges_dir = '/media/martina/Data/School/CTU/thesis/mars_edges'

    # shutil.rmtree(os.path.join(output_dir, 'joints_edges'))
    # shutil.rmtree(os.path.join(output_dir, 'joints'))
    # shutil.rmtree(os.path.join(output_dir, 'key_points'))
    #
    # os.mkdir(os.path.join(output_dir, 'joints_edges'))
    # os.mkdir(os.path.join(output_dir, 'joints'))
    # os.mkdir(os.path.join(output_dir, 'key_points'))

    all_identities = os.listdir(input_dir)
    processed_identities = os.listdir(os.path.join(output_dir, 'joints_edges'))
    # [all_identities.remove(identity) for identity in processed_identities]
    all_identities.sort()
    p = PP()
    for identity in all_identities:
        start = timeit.timeit()
        os.mkdir(os.path.join(output_dir, 'joints_edges_original', identity))
        images = os.listdir(os.path.join(input_dir, identity))
        images.sort()

        for image in images:
            p.process_img(image, identity, output_dir, input_dir, edges_dir)
        end = timeit.timeit()
        print("[INFO] time to load one identity : " + str(end - start))
