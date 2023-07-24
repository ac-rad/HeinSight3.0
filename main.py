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
from temporal_attention_rcnn import AttentionTrainer, AttentionPredictor
import cv2
import tqdm
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
from buffer import RollingAverageSmoothing

classes = ["Homogeneous Reaction", "Heterogeneous Reaction", "Residue", "Empty", "Solid", "StirBar"]
colors = [(189/255.0, 16/255.0, 224/255.0), (245/255.0, 166/255.0, 35/255.0), (110/255.0, 226/255.0, 105/255.0), (248/255.0, 231/255.0, 28/255.0), (0/255.0, 60/255.0, 255/255.0), (60/255.0, 60/255.0, 60/255.0)]


solid_classes = ["Residue", "Solid", "StirBar"]
solid_colors = [(248/255.0, 231/255.0, 28/255.0), (0/255.0, 60/255.0, 255/255.0), (110/255.0, 226/255.0, 105/255.0)]
liquid_classes = ["Homogeneous Reaction", "Heterogeneous Reaction", "Empty", "Cap"]
liquid_colors = [(189/255.0, 16/255.0, 224/255.0), (245/255.0, 166/255.0, 35/255.0), (120/255.0, 120/255.0, 120/255.0), (60/255.0, 60/255.0, 60/255.0)]


LOG = init_logger.get_logger('instance_seg.log')
MetadataCatalog.get("LLDataset").set(
    thing_classes=["Homogeneous Reaction", "Heterogeneous Reaction", "Residue", "Empty", "Solid", "StirBar"]).set(
    thing_colors=[(248, 231, 28), (0, 60, 255), (189, 16, 224), (60, 60, 60), (245, 166, 35), (110, 226, 105)])
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


def initialize_yolo():
    liquid_model = torch.hub.load('./yolov5', 'custom', path='./liquid/best.pt', source='local')
    solid_model = torch.hub.load('./yolov5', 'custom', path='./solid/best.pt', source='local')
    return liquid_model, solid_model


def initialize_rcnn():
    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1
    cfg.MODEL.WEIGHTS = "output_60ctx_channel_attn_blk_rem/model_final.pth"  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = "liquid/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    cfg.TEST.DETECTIONS_PER_IMAGE = 4
    cfg.MODEL.DEVICE = "cuda"
    cfg.INPUT.MAX_SIZE_TRAIN = 768
    cfg.INPUT.MIN_SIZE_TRAIN = 384
    cfg.INPUT.MAX_SIZE_TEST = 768
    cfg.INPUT.MIN_SIZE_TEST = 384
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 100
    liquid_predictor = AttentionPredictor(cfg)
    # liquid_predictor = DefaultPredictor(cfg)
    cfg2 = get_cfg()
    cfg2.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg2.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg2.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1
    cfg2.MODEL.WEIGHTS = "solid/model_final.pth"  # path to the model we just trained
    cfg2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    cfg2.TEST.DETECTIONS_PER_IMAGE = 10
    cfg2.MODEL.DEVICE = "cuda"
    solid_predictor = DefaultPredictor(cfg2)
    return liquid_predictor, solid_predictor


def eval_yolo(im, boxes, liquid_predictor, scale=1.0):
    v_cropped = Visualizer(im,
                           metadata=LLDatasetMetadate,
                           scale=1.0,
                           instance_mode=ColorMode.SEGMENTATION
                           # remove the colors of unsegmented pixels. This option is only available for segmentation models
                           )
    for box in boxes:
        x, y, w, h, = box
        x, y, w, h = int(x*scale), int(y*scale), int(w*scale), int(h*scale)
        seg = im[y:y+h, x:x+w]
        results = liquid_predictor(seg, size=640)
        for boxp in results.xyxyn[0].to('cpu'):
            classp = int(boxp[5].item())
            scorep = boxp[4].item()
            boxp = boxp[:4]*torch.Tensor([w, h, w, h]).to('cpu') + torch.Tensor([x, y, x, y]).to('cpu')
            v_cropped.draw_box(boxp, edge_color=liquid_colors[classp])
            v_cropped.draw_text(f'{liquid_classes[classp]}, {scorep:.2f}', tuple(boxp[:2].numpy()),
                                                        color=liquid_colors[classp])
    return v_uncropped.get_output().get_image(), v_cropped.get_output().get_image()



def eval(im, boxes, liquid_predictor, solid_predictor, scale=1.0, context=[], rolling_buffer=None):
    # solid_uncropped_outputs = solid_predictor(im)
    # liquid_uncropped_outputs = liquid_predictor(im)
    v_cropped = Visualizer(im,
                             metadata=LLDatasetMetadate,
                             scale=1.0,
                             instance_mode=ColorMode.SEGMENTATION
                             # remove the colors of unsegmented pixels. This option is only available for segmentation models
                             )
    v_uncropped = Visualizer(im,
                             metadata=LLDatasetMetadate,
                             scale=1.0,
                             instance_mode=ColorMode.SEGMENTATION
                             # remove the colors of unsegmented pixels. This option is only available for segmentation models
                             )
    for box in boxes:
        x, y, w, h, = box
        x, y, w, h = int(x*scale), int(y*scale), int(w*scale), int(h*scale)
        input_context = []
        for frame in context:
            seg = im[y:y+h, x:x+w]
            seg = cv2.resize(seg, (384, 768))
            input_context.append(seg.copy())
        # cap_ratio = 0.0
        # x, y = x, int(y + h*cap_ratio)
        # h, w = int((1-cap_ratio)*h), w
        scale_y, scale_x = h/768.0, w/384.0
        seg = im[y:y+h, x:x+w]
        # seg = cv2.resize(seg, (384, 768))
        # solid_outputs = solid_predictor(seg)
        seg = im[y:y+h, x:x+w]
        seg = cv2.resize(seg, (384, 768))
        liquid_outputs = liquid_predictor(seg, input_context)
        # liquid_outputs = liquid_predictor(seg)
        # i = 0
        # if i==0:
        #     cv2.imwrite('./tmp.jpg', seg)
        #     return
        # for boxp in solid_outputs["instances"].pred_boxes.to('cpu'):
        #     boxp = boxp + torch.Tensor([x, y, x, y]).to('cpu')
        #     v_cropped.draw_box(boxp, edge_color=solid_colors[solid_outputs["instances"].pred_classes[i]])
        #     v_cropped.draw_text(f'{solid_classes[solid_outputs["instances"].pred_classes[i]]}, {solid_outputs["instances"].scores[i]:.2f}', tuple(boxp[:2].numpy()),
        #                           color=solid_colors[solid_outputs["instances"].pred_classes[i]])
        #     i += 1
        i = 0
        if rolling_buffer:
            boxes = rolling_buffer.process(liquid_outputs["instances"])

        for boxp in liquid_outputs["instances"].pred_boxes.to('cpu'):
            # if liquid_classes[liquid_outputs["instances"].pred_classes[i]] == "Empty":
            #     i += 1
            #     continue
            boxp = boxp*torch.Tensor([scale_x, scale_y, scale_x, scale_y]).to('cpu') + torch.Tensor([x, y, x, y]).to('cpu')
            v_cropped.draw_box(boxp, edge_color=liquid_colors[liquid_outputs["instances"].pred_classes[i]])
            v_cropped.draw_text(f'{liquid_classes[liquid_outputs["instances"].pred_classes[i]]}, {liquid_outputs["instances"].scores[i]:.2f}', tuple(boxp[:2].numpy()),
                                color=liquid_colors[liquid_outputs["instances"].pred_classes[i]])
            i += 1

    # i = 0
    # for box in solid_uncropped_outputs["instances"].pred_boxes.to('cpu'):
    #     v_uncropped.draw_box(box, edge_color=solid_colors[solid_uncropped_outputs["instances"].pred_classes[i]])
    #     v_uncropped.draw_text(str(solid_classes[solid_uncropped_outputs["instances"].pred_classes[i]]), tuple(box[:2].numpy()),
    #                 color=solid_colors[solid_uncropped_outputs["instances"].pred_classes[i]])
    #     i += 1
    # i = 0
    # for box in liquid_uncropped_outputs["instances"].pred_boxes.to('cpu'):
    #     v_uncropped.draw_box(box, edge_color=liquid_colors[liquid_uncropped_outputs["instances"].pred_classes[i]])
    #     v_uncropped.draw_text(str(liquid_classes[liquid_uncropped_outputs["instances"].pred_classes[i]]),
    #                           tuple(box[:2].numpy()),
    #                           color=liquid_colors[liquid_uncropped_outputs["instances"].pred_classes[i]])
    #     i += 1
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
    LOG.info(f'segment complete, masks found: {len(vial_bbox)} {vial_bbox}')
    LOG.info(f'Initializing HeinSight2.0')
    liquid_predictor, solid_predictor = initialize_yolo()
    rolling_buffer = RollingAverageSmoothing(smoothness=0.7, period=60, num_predictions=4)
    # vial_bbox = [[425, 45, 248, 680], [745, 29, 229, 700], [1080, 37, 238, 687], [1427, 36, 257, 681]]
    # vial_bbox = [[65, 195, 296, 610], [437, 180, 291, 600], [871, 166, 253, 582], [1180, 160, 255, 578], [1523, 159, 291, 565]]
    vial_bbox = [[648, 30, 262, 713], [1005, 43, 265, 694], [1325, 35, 270, 705]]
    # vial_bbox = [[303, 32, 303, 719], [702, 33, 273, 719], [984, 37, 268, 709], [1325, 37, 288, 707]]
    # vial_bbox = [[447, 3, 156, 467], [764, 3, 157, 461]]
    # vial_bbox = [[598, 128, 109, 312]]
    # vial_bbox = [[597, 106, 118, 298]]
    # vial_bbox = [vial_bbox[-1]]
    # vial_bbox = [[616, 0, 107, 319]]
    writer1 = imageio.get_writer(f"./output/insseg/uncrop_{input_image_name.split('.')[0]}.mp4", fps=60)
    writer2 = imageio.get_writer(f"./output/insseg/crop_{input_image_name.split('.')[0]}.mp4", fps=60)
    cap = cv2.VideoCapture(input_image_path)
    LOG.info(f'Analysing Video')
    pbar = tqdm.tqdm(total=3800)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    num_context = 0
    frame_rate = int(round(fps))//6
    frame_drain = frame_rate
    max_buff = (num_context)*frame_rate
    min_buff = frame_drain
    buff = []
    printed = False
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            scale = 1920.0/1920.0
            resized_frame = cv2.resize(frame, (1920, 1080))
            if len(buff)<max_buff:
                buff.append(resized_frame.copy())
            if len(buff)<min_buff:
                buff.append(resized_frame.copy())
                writer1.append_data(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
                writer2.append_data(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
                pbar.update(1)
                continue
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # context = buff[:-frame_rate+1:frame_rate]
            context = []
            if len(buff)<max_buff:
                assert len(context)<num_context and len(context)!=0
            else:
                assert len(context) == num_context
            if len(context)==num_context and not printed:
                LOG.info(f"Max context reached {len(context), len(buff)}")
                printed = True
            # uncrop_im, crop_im = eval(resized_frame, vial_bbox, liquid_predictor, solid_predictor, scale=scale, context=context)
            uncrop_im, crop_im = eval_yolo(resized_frame, vial_bbox, liquid_predictor, scale=scale)
            writer1.append_data(cv2.cvtColor(uncrop_im, cv2.COLOR_BGR2RGB))
            writer2.append_data(cv2.cvtColor(crop_im, cv2.COLOR_BGR2RGB))
            if len(buff)>max_buff:
                buff.pop(0)
            pbar.update(1)
        else:
            cap.release()
            break
    writer1.close()
    writer2.close()
    pbar.close()
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
