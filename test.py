# System libs
import os
import time

# Numerical libs
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

# Our libs
from data.sist_line import SISTLine
import data.transforms as tf
from models.lsd_test import LSDTestModule
from utils import AverageMeter, graph2line, draw_lines, draw_jucntions, plot_from_files

# tensorboard
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

import fire
import cv2


class LSD(object):
    def __init__(
            self,
            # exp params
            exp_name="u50_block",
            # arch params
            backbone="resnet50",
            backbone_kwargs={},
            dim_embedding=256,
            feature_spatial_scale=0.25,
            max_junctions=512,
            junction_pooling_threshold=0.2,
            junc_pooling_size=15,
            block_inference_size=64,
            # data params
            img_size=416,
            gpus=[0,],
            resume_epoch="latest",
            # vis params
            vis_junc_th=0.3,
            vis_line_th=0.3
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(c) for c in gpus)

        self.is_cuda = bool(gpus)

        self.model = LSDTestModule(
            backbone=backbone,
            dim_embedding=dim_embedding,
            backbone_kwargs=backbone_kwargs,
            junction_pooling_threshold=junction_pooling_threshold,
            max_junctions=max_junctions,
            feature_spatial_scale=feature_spatial_scale,
            junction_pooling_size=junc_pooling_size,
        )

        self.exp_name = exp_name
        os.makedirs(os.path.join("log", exp_name), exist_ok=True)
        os.makedirs(os.path.join("ckpt", exp_name), exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join("log", exp_name))

        # checkpoints
        self.states = dict(
            last_epoch=-1,
            elapsed_time=0,
            state_dict=None
        )

        if resume_epoch and os.path.isfile(os.path.join("ckpt", exp_name, f"train_states_{resume_epoch}.pth")):
            states = torch.load(
                os.path.join("ckpt", exp_name, f"train_states_{resume_epoch}.pth"))
            print(f"resume traning from epoch {states['last_epoch']}")
            self.model.load_state_dict(states["state_dict"])
            self.states.update(states)
            self.last_epoch = states['last_epoch']
        else:
            self.last_epoch = 0

        self.vis_junc_th = vis_junc_th
        self.vis_line_th = vis_line_th
        self.block_size = block_inference_size
        self.max_junctions = max_junctions
        self.img_size = img_size
        
    def end(self):
        self.writer.close()
        return "command queue finished."

    def test(self, path_to_image):
        # fn to save image with lines and points and coordinate files
        fn = f'{path_to_image.rsplit(".", 1)[0]}_epoch{self.last_epoch}'

        # main loop
        torch.set_grad_enabled(False)
        print(f"test for image: {path_to_image}", flush=True)

        if self.is_cuda:
            model = self.model.cuda().eval()
        else:
            model = self.model.eval()

        img = cv2.imread(path_to_image)

        # Use the original shape so we can resize the results in the end
        original_size = img.shape[:2]
        original_width = original_size[1]
        original_height = original_size[0]
        width_scale = original_width / self.img_size
        height_scale = original_height / self.img_size
        self.scales = (width_scale, height_scale)

        img = cv2.resize(img, (self.img_size, self.img_size))
        img_reverse = img[..., [2, 1, 0]]
        img = torch.from_numpy(img_reverse).float().permute(2, 0, 1).unsqueeze(0)

        if self.is_cuda:
            img = img.cuda()

        # measure elapsed time
        junc_pred, heatmap_pred, adj_mtx_pred = model(img)

        # visualize eval
        img = img.cpu().numpy()
        junctions_pred = junc_pred.cpu().numpy()
        adj_mtx = adj_mtx_pred.cpu().numpy()

        img_with_junc = draw_jucntions(img, junctions_pred, save_junction=fn, scales=self.scales)
        img_with_junc = img_with_junc[0].numpy()[None]
        img_with_junc = img_with_junc[:, ::-1, :, :]
        lines_pred, score_pred = graph2line(junctions_pred, adj_mtx)
        draw_lines(img_with_junc, lines_pred, score_pred, 
                    save_lines=fn, scales=self.scales)[0]

        # Txt files are created, use matplotlib to save the original image with the coordinates
        #NOTE: change path_to_image to path_to_image.replace('_enhanced', ''),
        #if you want to show on original without "drawn" edges
        img = cv2.imread(path_to_image)
        plot_from_files(fn+'_junctions.txt', fn+'_lines.txt', path_to_image, fn+'_annotated.png')


if __name__ == "__main__":
    fire.Fire(LSD)
    # trainer = LSDTrainer().train(lr=1.)
