#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-04-11 下午2:15
# @Author  : MaybeShewill-CV
# @Site    :
# @File    : sam_clip_text_seg.py
# @IDE: PyCharm Community Edition
"""
instance segmentation image with sam and clip with text prompts
"""
import os
import os.path as ops
import argparse
from PIL import Image
import imageio
import cv2
import torch
from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils
from models import build_sam_clip_text_ins_segmentor
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog

classes = ["Homogeneous Reaction", "Heterogeneous Reaction", "Residue", "Empty"]
colors = [(189/255.0, 16/255.0, 224/255.0), (245/255.0, 166/255.0, 35/255.0), (110/255.0, 226/255.0, 105/255.0), (248/255.0, 231/255.0, 28/255.0), (0/255.0, 60/255.0, 255/255.0), (0/255.0, 60/255.0, 255/255.0)]

LOG = init_logger.get_logger('instance_seg.log')
MetadataCatalog.get("LLDataset").set(
    thing_classes=["Homogeneous Reaction", "Heterogeneous Reaction", "Residue", "Empty"]).set(
    thing_colors=[(189, 16, 224), (245, 166, 35), (110, 226, 105), (248, 231, 28), (0, 60, 255),
                  (200, 200, 200)])
LLDatasetMetadate = MetadataCatalog.get("LLDataset")


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_path', type=str, default='./data/test_bear.jpg', required=True)
    parser.add_argument('--insseg_cfg_path', type=str, default='./config/insseg.yaml')
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--cls_score_thresh', type=float, default=None)
    parser.add_argument('--save_dir', type=str, default='./output/insseg')
    parser.add_argument('--use_text_prefix', action='store_true')

    return parser.parse_args()


def main():
    """

    :return:
    """
    # init args
    args = init_args()
    input_image_path = args.input_image_path
    input_image_name = ops.split(input_image_path)[1]
    if not ops.exists(input_image_path):
        LOG.error('input image path: {:s} not exists'.format(input_image_path))
        return
    insseg_cfg_path = args.insseg_cfg_path
    if not ops.exists(insseg_cfg_path):
        LOG.error('input innseg cfg path: {:s} not exists'.format(insseg_cfg_path))
        return
    insseg_cfg = parse_config_utils.Config(config_path=insseg_cfg_path)
    if args.text is not None:
        unique_labels = args.text.split(',')
    else:
        unique_labels = None
    if args.cls_score_thresh is not None:
        insseg_cfg.INS_SEG.CLS_SCORE_THRESH = args.cls_score_thresh
    use_text_prefix = True if args.use_text_prefix else False

    # init cluster
    LOG.info('Start initializing instance segmentor ...')
    segmentor = build_sam_clip_text_ins_segmentor(cfg=insseg_cfg)
    LOG.info('Segmentor initialized complete')
    LOG.info('Start to segment input image ...')
    ret = segmentor.seg_image(input_image_path, unique_label=unique_labels, use_text_prefix=use_text_prefix)
    LOG.info('segment complete')

    # save cluster result
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    ori_image_save_path = ops.join(save_dir, input_image_name)
    cv2.imwrite(ori_image_save_path, ret['source'])
    mask_save_path = ops.join(save_dir, '{:s}_insseg_mask.png'.format(input_image_name.split('.')[0]))
    cv2.imwrite(mask_save_path, ret['ins_seg_mask'])
    mask_add_save_path = ops.join(save_dir, '{:s}_insseg_add.png'.format(input_image_name.split('.')[0]))
    cv2.imwrite(mask_add_save_path, ret['ins_seg_add'])

    LOG.info('save segment and cluster result into {:s}'.format(save_dir))

    return

def initialize_rcnn():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1
    cfg.MODEL.WEIGHTS = "model_final.pth"  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    cfg.TEST.DETECTIONS_PER_IMAGE = 4
    cfg.MODEL.DEVICE = "cuda"
    predictor = DefaultPredictor(cfg)
    return predictor


def eval(im, boxes, predictor):
    uncropped_outputs = predictor(im)
    v_cropped = Visualizer(im,
                             metadata=LLDatasetMetadate,
                             scale=.6,
                             instance_mode=ColorMode.SEGMENTATION
                             # remove the colors of unsegmented pixels. This option is only available for segmentation models
                             )
    v_uncropped = Visualizer(im,
                             metadata=LLDatasetMetadate,
                             scale=.6,
                             instance_mode=ColorMode.SEGMENTATION
                             # remove the colors of unsegmented pixels. This option is only available for segmentation models
                             )
    for box in boxes:
        x, y, w, h, = box
        seg = im[y:y+h, x:x+w]
        outputs = predictor(seg)
        i = 0
        for boxp in outputs["instances"].pred_boxes.to('cpu'):
            boxp = boxp + torch.Tensor([x, y, x, y]).to('cpu')
            v_cropped.draw_box(boxp, edge_color=colors[outputs["instances"].pred_classes[i]])
            v_cropped.draw_text(str(classes[outputs["instances"].pred_classes[i]]), tuple(boxp[:2].numpy()),
                                  color=colors[outputs["instances"].pred_classes[i]])
            i += 1

    i = 0
    for box in uncropped_outputs["instances"].pred_boxes.to('cpu'):
        v_uncropped.draw_box(box, edge_color=colors[uncropped_outputs["instances"].pred_classes[i]])
        v_uncropped.draw_text(str(classes[uncropped_outputs["instances"].pred_classes[i]]), tuple(box[:2].numpy()),
                    color=colors[uncropped_outputs["instances"].pred_classes[i]])
        i += 1
    return v_uncropped.get_output().get_image(), v_cropped.get_output().get_image()


def segment_video():
    args = init_args()
    input_image_path = args.input_image_path
    input_image_name = ops.split(input_image_path)[1]
    if not ops.exists(input_image_path):
        LOG.error('input video path: {:s} not exists'.format(input_image_path))
        return
    insseg_cfg_path = args.insseg_cfg_path
    if not ops.exists(insseg_cfg_path):
        LOG.error('input innseg cfg path: {:s} not exists'.format(insseg_cfg_path))
        return
    insseg_cfg = parse_config_utils.Config(config_path=insseg_cfg_path)
    if args.text is not None:
        unique_labels = args.text.split(',')
    else:
        unique_labels = None
    if args.cls_score_thresh is not None:
        insseg_cfg.INS_SEG.CLS_SCORE_THRESH = args.cls_score_thresh
    use_text_prefix = True if args.use_text_prefix else False
    cap = cv2.VideoCapture(input_image_path)
    if cap.isOpened():
        ret, frame = cap.read()
        resized_frame = cv2.resize(frame, (1280, 720))
        cv2.imwrite(f"./data/test_images/{input_image_name.split('.')[0]}_one_frame_tmp.jpg", resized_frame)
        cap.release()

    # init cluster
    LOG.info('Start initializing instance segmentor ...')
    segmentor = build_sam_clip_text_ins_segmentor(cfg=insseg_cfg)
    LOG.info('Segmentor initialized complete')
    LOG.info('Start to segment input image ...')
    ret = segmentor.seg_image(f"./data/test_images/{input_image_name.split('.')[0]}_one_frame_tmp.jpg", unique_label=unique_labels, use_text_prefix=use_text_prefix)
    masks = ret['raw_masks']
    cv2.imwrite(f"./output/insseg/{input_image_name.split('.')[0]}_one_frame_tmp.jpg", ret['ins_seg_add'])
    vial_bbox = []
    for i in range(len(masks)):
        if ret["bboxes_names"][i]=="vial":
            vial_bbox.append(ret["bbox"][i])
    LOG.info(f'segment complete, masks found: {len(vial_bbox)}')
    LOG.info(f'Initializing HeinSight2.0')
    predictor = initialize_rcnn()

    writer1 = imageio.get_writer(f"./output/insseg/uncrop_{input_image_name.split('.')[0]}.mp4", fps=60)
    writer2 = imageio.get_writer(f"./output/insseg/crop_{input_image_name.split('.')[0]}.mp4", fps=60)
    cap = cv2.VideoCapture(input_image_path)
    LOG.info(f'Analysing Video')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame, (1280, 720))
            uncrop_im, crop_im = eval(resized_frame, vial_bbox, predictor)
            writer1.append_data(uncrop_im)
            writer2.append_data(crop_im)
        else:
            cap.release()
            break
    writer1.close()
    writer2.close()
    LOG.info(f'Videos saved at ./output/insseg/')
    # save cluster result
    # save_dir = args.save_dir
    # os.makedirs(save_dir, exist_ok=True)
    # ori_image_save_path = ops.join(save_dir, input_image_name)
    # cv2.imwrite(ori_image_save_path, ret['source'])
    # mask_save_path = ops.join(save_dir, '{:s}_insseg_mask.png'.format(input_image_name.split('.')[0]))
    # cv2.imwrite(mask_save_path, ret['ins_seg_mask'])
    # mask_add_save_path = ops.join(save_dir, '{:s}_insseg_add.png'.format(input_image_name.split('.')[0]))
    # cv2.imwrite(mask_add_save_path, ret['ins_seg_add'])
    #
    # LOG.info('save segment and cluster result into {:s}'.format(save_dir))

    return

if __name__ == '__main__':
    """
    main func
    """
    segment_video()
