# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 2019
Modified on Fri Mar 17 2023

@authors: Keshik, Ashish

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
from PIL import Image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from dataset import ImageFolder

import numpy as np
import pandas as pd
import torch
import yaml

def setup_logging(logging_config_path: str = 'logging.yaml', default_level: int = logging.INFO) -> None:
    if os.path.exists(logging_config_path):
        with open(logging_config_path, 'rt') as file:
            config = yaml.safe_load(file.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def get_filename_without_ext(path: str) -> str:
    filename_with_ext = os.path.basename(path)
    list = os.path.splitext(filename_with_ext)
    return list[0]

def generate_dictionary_for_predicted_object(sample: str, x1: int, y1: int, x2: int, y2: int, cls_pred: int, class_conf: float) -> dict:
    result = {}

    result['label'] = cls_pred
    result['score'] = class_conf
    result['bbox_xmin'] = x1
    result['bbox_ymin'] = y1
    result['bbox_xmax'] = x2
    result['bbox_ymax'] = y2

    result['truncated'] = 0.
    result['occluded'] = 0
    result['alpha'] = 0.
    result['dim_height'] = 0.
    result['dim_width'] = 0.
    result['dim_length'] = 0.
    result['loc_x'] = 0.
    result['loc_y'] = 0.
    result['loc_z'] = 0.
    result['rotation_y'] = 0.

    return result

def generate_dataframe_for_predicted_objects_in_sample(sample: str, detections: list) -> pd.DataFrame:
    columnNames=['label', 'truncated', 'occluded', 'alpha', 'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax', 'dim_height', 'dim_width', 'dim_length', 'loc_x', 'loc_y', 'loc_z', 'rotation_y', 'score' ]
    return pd.DataFrame(data=detections, columns=columnNames)

def write_predicted_labels_to_file(pathToOutputPredictedLabelsDirectory: str, sample: str, predictedLabelsDataFrame: pd.DataFrame) -> None:
    logger = logging.getLogger(__name__)
    outputPredictedLabelsFileName = "{sample}.{extension}".format(sample = sample, extension = "txt")
    outputPredictedLabelsFilePath = os.path.join(pathToOutputPredictedLabelsDirectory, outputPredictedLabelsFileName)

    columnOrder =['label', 'truncated', 'occluded', 'alpha', 'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax', 'dim_height', 'dim_width', 'dim_length', 'loc_x', 'loc_y', 'loc_z', 'rotation_y', 'score' ]
    predictedLabelsDataFrame[columnOrder].to_csv(outputPredictedLabelsFilePath, columns=columnOrder, index=False, header=True, sep=" ")
    logger.info('For sample %s, saved output annotated image to %s' % (sample, outputPredictedLabelsFilePath))

def detect(
        pathToModelWeights: str = '../checkpoints/best_weights_kitti.pth',
        pathToConfig: str = '../config/yolov3-kitti.cfg',
        pathToClasses: str = '../data/names.txt',
        pathToInputImages: str = '../data/samples/',
        pathToOutputPredictedLabelsDirectory: str = '../output/annotations/'):
    """
        Script to run inference on sample images. It will store all the inference results in /output directory (relative to repo root)
        
        Args
            kitti_weights: Path of weights
            config_path: Yolo configuration file path
            class_path: Path of class names txt file
            
    """
    logger = logging.getLogger(__name__)

    if not os.path.exists(pathToOutputPredictedLabelsDirectory):
        logger.info("Creating output directory: %s" % pathToOutputPredictedLabelsDirectory)
        os.makedirs(pathToOutputPredictedLabelsDirectory, exist_ok=True)

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

    logger.info('Number of images detected by the data loader: %d' % len(dataloader) )
    logger.info('Starting object detection')

    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in tqdm(enumerate(dataloader), desc="Inference: ", unit="batches", total=len(dataloader)):
        logger.debug("Batch: %d, #images: %d" % (batch_i, len(input_imgs)))

        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, 80, 0.8, 0.4)
            logger.debug("For batch %d, detections are below." % batch_i)
            logger.debug(detections)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        logger.debug('For batch %d, inference time is %s' % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    logger.info('Finished object detection')

    # Iterate through the detections and save them to the output directory 
    logger.debug('Computing rescaled predictions')
    predictedLabelsDictionary = {}
    for img_i, (path, detections) in tqdm(enumerate(zip(imgs, img_detections)), desc="Rescaling predictions: ", unit="images", total=len(imgs)):
        predictions = []

        sample = get_filename_without_ext(path)
        logger.info("Index: (%d), sample: %s, image path: '%s'" % (img_i, sample, path))

        img = np.array(Image.open(path))
        kitti_img_size = 416
        
        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (kitti_img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (kitti_img_size / max(img.shape))

        # Image height and width after padding is removed
        unpad_h = kitti_img_size - pad_y
        unpad_w = kitti_img_size - pad_x

        # Draw bounding boxes and labels of detections
        if detections is not None:
            logger.debug("Number of detections: {0}".format(detections.shape[0]))
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # Rescale coordinates to original dimensions
                box_h = int(((y2 - y1) / unpad_h) * (img.shape[0]))
                box_w = int(((x2 - x1) / unpad_w) * (img.shape[1]))
                y1 = int(((y1 - pad_y // 2) / unpad_h) * (img.shape[0]))
                x1 = int(((x1 - pad_x // 2) / unpad_w) * (img.shape[1]))
                logger.debug('Class: %s, Conf: %.5f, Bounding box: (%d, %d), (%d, %d)' % (classes[int(cls_pred)], cls_conf.item(), x1, y1, x1 + box_w, y1 + box_h))
                prediction = generate_dictionary_for_predicted_object(sample=sample, x1=x1, y1=y1, x2=x1 + box_w, y2=y1 + box_h, cls_pred=classes[int(cls_pred)], class_conf=cls_conf.item())
                predictions.append(prediction)

        # Generate dataframe from predictions
        predictedLabelsDataFrame = generate_dataframe_for_predicted_objects_in_sample(sample, predictions) 
        logger.debug('Predicted labels for sample %s are below.' % sample)
        logger.debug(predictedLabelsDataFrame)
        predictedLabelsDictionary[sample] = predictedLabelsDataFrame

    logger.info('Saving predictions to directory %s' % pathToOutputPredictedLabelsDirectory)
    for sample, predictedLabelsDataFrame in tqdm(predictedLabelsDictionary.items(), desc="Saving predictions: ", unit="samples", total=len(predictedLabelsDictionary)):
        write_predicted_labels_to_file(pathToOutputPredictedLabelsDirectory, sample, predictedLabelsDataFrame)

def main():
    # Logging Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info('Started')

    # Run the model
    torch.multiprocessing.freeze_support()
    detect()

    logger.info('Finished')

if __name__ == '__main__':
    main()