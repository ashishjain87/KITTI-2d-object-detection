# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:47:01 2019

@author: Keshik, Ashish

Source
    https://github.com/packyan/PyTorch-YOLOv3-kitti
"""

from __future__ import division

import os
import datetime
import logging
import logging.config
import time

from model import Darknet
from utils import non_max_suppression, load_classes

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import ImageFolder
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import random

def get_filename_without_ext(path: str) -> str:
    filename_with_ext = os.path.basename(path)
    list = os.path.splitext(filename_with_ext)
    return list[0]

def detect(
        pathToModelWeights: str = '../checkpoints/best_weights_kitti.pth',
        pathToConfig: str = '../config/yolov3-kitti.cfg',
        pathToClasses: str = '../data/names.txt',
        pathToInputImages: str = '../data/samples/',
        pathToOutputAnnotationsDirectory: str = '../output/annotations/',
        pathToOutputImagesDirectory: str = '../output/'):
    """
        Script to run inference on sample images. It will store all the inference results in /output directory (relative to repo root)
        
        Args
            kitti_weights: Path of weights
            config_path: Yolo configuration file path
            class_path: Path of class names txt file
            
    """
    logger = logging.getLogger(__name__)


    if not os.path.exists(pathToOutputImagesDirectory):
        logger.info("Creating output directory: %s" % pathToOutputImagesDirectory)
        os.makedirs(pathToOutputImagesDirectory, exist_ok=True)

    if not os.path.exists(pathToOutputAnnotationsDirectory):
        logger.info("Creating output directory: %s" % pathToOutputAnnotationsDirectory)
        os.makedirs(pathToOutputAnnotationsDirectory, exist_ok=True)

    # Set up model
    model = Darknet(pathToConfig, img_size=416)
    model.load_weights(pathToModelWeights)
    logger.debug("Model loaded successfully. Loaded weights from: %s" % pathToModelWeights)

    cuda = torch.cuda.is_available()
    if cuda:
        logger.info("Cuda available for inference: %s" % cuda)
        model.cuda()

    model.eval() # Set in evaluation mode
    logger.debug("Model set in evaluation mode successfully")

    dataloader = DataLoader(ImageFolder(pathToInputImages, img_size=416),
                            batch_size=2, shuffle=False, num_workers=0)

    classes = load_classes(pathToClasses) # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    imgs = []           # Stores image paths
    img_detections = [] # Stores detections for each image index

    logger.info('data size : %d' % len(dataloader) )
    logger.info('Performing object detection:')
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, 80, 0.8, 0.4)
            logger.info(detections)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        logger.info('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    #cmap = plt.get_cmap('tab20b')
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    logger.info('Saving images:')
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        logger.info("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        kitti_img_size = 416
        
        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (kitti_img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (kitti_img_size / max(img.shape))
        # Image height and width after padding is removed
        unpad_h = kitti_img_size - pad_y
        unpad_w = kitti_img_size - pad_x

        # Draw bounding boxes and labels of detections
        if detections is not None:
            logger.debug("Type of detections: %s" % type(detections))
            logger.debug("Number of detections: %s" % detections.size())
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                logger.debug('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
                # Rescale coordinates to original dimensions
                box_h = int(((y2 - y1) / unpad_h) * (img.shape[0]))
                box_w = int(((x2 - x1) / unpad_w) * (img.shape[1]) )
                y1 = int(((y1 - pad_y // 2) / unpad_h) * (img.shape[0]))
                x1 = int(((x1 - pad_x // 2) / unpad_w) * (img.shape[1]))
                logger.debug('\t+ Class: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item())) 
                logger.debug('\t+ Rescaled bounding box coordinates: (%d, %d), (%d, %d)' % (x1, y1, x1 + box_w, y1 + box_h))

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                        edgecolor=color,
                                        facecolor='none')
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(x1, y1-30, s=classes[int(cls_pred)]+' '+ str('%.4f'%cls_conf.item()), color='white', verticalalignment='top',
                        bbox={'color': color, 'pad': 0})

        # Save generated image with detections
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())

        sample = get_filename_without_ext(path)
        outputImageFileName = "{sample}.{extension}".format(sample = sample, extension = "png")
        outputImageFilePath = os.path.join(pathToOutputImagesDirectory, outputImageFileName)
        plt.savefig(outputImageFilePath, bbox_inches='tight', pad_inches=0.0)
        plt.close()

        logger.info('For sample %s, saved output annotated image to %s' % (sample, outputImageFilePath))


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    detect()
