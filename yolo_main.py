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
import tqdm
import torch
import time
from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog

classes = ["Homogeneous Reaction", "Heterogeneous Reaction", "Residue", "Empty", "Solid", "StirBar"]
colors = [(189/255.0, 16/255.0, 224/255.0), (245/255.0, 166/255.0, 35/255.0), (110/255.0, 226/255.0, 105/255.0), (248/255.0, 231/255.0, 28/255.0), (0/255.0, 60/255.0, 255/255.0), (60/255.0, 60/255.0, 60/255.0)]


solid_classes = ["Residue", "Solid", "StirBar"]
solid_colors = [(248/255.0, 231/255.0, 28/255.0), (0/255.0, 60/255.0, 255/255.0), (110/255.0, 226/255.0, 105/255.0)]
liquid_classes = ["Homogeneous Reaction", "Heterogeneous Reaction", "Empty", "Cap"]
liquid_colors = [(189, 16, 224), (245, 166, 35), (120, 120, 120), (60, 60, 60)]


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
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cls_score_thresh', type=float, default=None)
    parser.add_argument('--save_dir', type=str, default='./output/insseg')
    parser.add_argument('--use_text_prefix', action='store_true')

    return parser.parse_args()

def initialize_vial_detector():
    model = torch.hub.load('./yolov5', 'custom', path='./vial/best.pt', source='local')
    return model

def initialize_yolo():
    liquid_model = torch.hub.load('./yolov5', 'custom', path='./liquid/best.pt', source='local')
    solid_model = torch.hub.load('./yolov5', 'custom', path='./solid/best.pt', source='local')
    return liquid_model, solid_model


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
        results = liquid_predictor(batch, size=640)
        for boxp in results.xyxyn[0].to('cpu'):
            classp = int(boxp[5].item())
            scorep = boxp[4].item()
            boxp = boxp[:4]*torch.Tensor([w, h, w, h]).to('cpu') + torch.Tensor([x, y, x, y]).to('cpu')
            v_cropped.draw_box(boxp, edge_color=liquid_colors[classp])
            v_cropped.draw_text(f'{liquid_classes[classp]}, {scorep:.2f}', tuple(boxp[:2].numpy()),
                                                        color=liquid_colors[classp])
    return v_cropped.get_output().get_image(), v_cropped.get_output().get_image()


def eval_yolo_batch(ims, boxes, liquid_predictor, scale=1.0):
    batch = []
    ret_cropped = []
    ret_uncropped = []
    # start = time.time()
    for im in ims:
        for box in boxes:
            x, y, w, h, = box
            x, y, w, h = int(x*scale), int(y*scale), int(w*scale), int(h*scale)
            seg = im[y:y+h, x:x+w]
            batch.append(seg.copy())
    # print(len(batch))
    # end = time.time()
    # LOG.info(f"Time to initialize batch {end-start}")
    # start = time.time()
    results = liquid_predictor(batch, size=640)
    # end = time.time()
    # LOG.info(f"Inference time {end-start}")
    for im_idx, im in enumerate(ims):
        # start = time.time()
        # v_cropped = Visualizer(im,
        #                        metadata=LLDatasetMetadate,
        #                        scale=1.0,
        #                        instance_mode=ColorMode.SEGMENTATION
        #                        # remove the colors of unsegmented pixels. This option is only available for segmentation models
        #                        )
        # end = time.time()
        # LOG.info(f"Time to initialize visualizer {end-start}")

        # start = time.time()
        for box_idx, box in enumerate(boxes):
            x, y, w, h, = box
            x, y, w, h = int(x*scale), int(y*scale), int(w*scale), int(h*scale)
            for boxp in results.xyxyn[im_idx*len(boxes)+box_idx].to('cpu'):
                classp = int(boxp[5].item())
                scorep = boxp[4].item()
                boxp = boxp[:4]*torch.Tensor([w, h, w, h]).to('cpu') + torch.Tensor([x, y, x, y]).to('cpu')
                im = cv2.rectangle(im, tuple(boxp[:2].numpy().astype(int)), tuple(boxp[2:].numpy().astype(int)), liquid_colors[classp], 1)
                im = cv2.putText(im, f'{liquid_classes[classp]}, {scorep:.2f}', tuple(boxp[:2].numpy().astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, liquid_colors[classp], 1)
                # v_cropped.draw_box(boxp, edge_color=liquid_colors[classp])
                # v_cropped.draw_text(f'{liquid_classes[classp]}, {scorep:.2f}', tuple(boxp[:2].numpy()), color=liquid_colors[classp])
        # end = time.time()
        # LOG.info(f"Time to create visualizations {end-start}")
        # start = time.time()
        ret_cropped.append(im)
        ret_uncropped.append(im)
        # end = time.time()
        # LOG.info(f"Time to return inference {end-start}")
    return ret_cropped, ret_uncropped


def segment_video():
    args = init_args()
    input_image_path = args.input_image_path
    input_image_name = ops.split(input_image_path)[1]
    if not ops.exists(input_image_path):
        LOG.error('input video path: {:s} not exists'.format(input_image_path))
        return
    LOG.info(f"Initializing vessel detector")
    VESSEL_THRESH = 0.8
    vial_detector = initialize_vial_detector()
    liquid_predictor, solid_predictor = initialize_yolo()
    cap = cv2.VideoCapture(input_image_path)
    vial_bbox =[]
    if cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = vial_detector(frame)
        results.save(save_dir='./output/insseg', exist_ok=True)
        mask = results.xyxyn[0][:, 4]>=VESSEL_THRESH
        indices = torch.nonzero(mask)
        results.xyxyn[0] = results.xyxyn[0][indices][0]
        vial_bbox = results.xyxyn[0][:, :4].to('cpu')*torch.tensor([1920.0, 1080.0, 1920.0, 1080.0]).to('cpu')
        vial_bbox[:, 2] = vial_bbox[:, 2] - vial_bbox[:, 0]
        vial_bbox[:, 3] = vial_bbox[:, 3] - vial_bbox[:, 1]
        vial_bbox = vial_bbox.to(torch.int32).tolist()
        # cv2.imwrite(f"./data/test_images/{input_image_name.split('.')[0]}_one_frame_tmp.jpg", results.plot())
        cap.release()
    assert vial_bbox!=[]
    LOG.info(f"Detected {len(vial_bbox)} vials at: {vial_bbox}")
    # init cluster
    LOG.info(f'Initializing HeinSight2.0')
    # liquid_predictor, solid_predictor = initialize_yolo()
    # vial_bbox = [[425, 45, 248, 680], [745, 29, 229, 700], [1080, 37, 238, 687], [1427, 36, 257, 681]]
    # vial_bbox = [[65, 195, 296, 610], [437, 180, 291, 600], [871, 166, 253, 582], [1180, 160, 255, 578], [1523, 159, 291, 565]]
    # vial_bbox = [[648, 30, 262, 713], [1005, 43, 265, 694], [1325, 35, 270, 705]]
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
    min_buff = 0*frame_drain
    buff = []
    printed = False
    batch = []
    batch_size=args.batch_size
    LOG.info(f"Batch size: {batch_size}")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            scale = 1920.0/1920.0
            frame_count += 1
            resized_frame = cv2.resize(frame, (1920, 1080))
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            batch.append(resized_frame)
            if frame_count%batch_size != 0:
                pbar.update(1)
                continue
            # if len(buff)<max_buff:
            #     buff.append(resized_frame.copy())
            # if len(buff)<min_buff:
            #     buff.append(resized_frame.copy())
            #     writer1.append_data(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            #     writer2.append_data(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            #     pbar.update(1)
            #     continue
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # context = buff[:-frame_rate+1:frame_rate]
            context = []
            # if len(buff)<max_buff:
            #     assert len(context)<num_context and len(context)!=0
            # else:
            #     assert len(context) == num_context
            if len(context)==num_context and not printed:
                LOG.info(f"Max context reached {len(context), len(buff)}")
                printed = True
            # uncrop_im, crop_im = eval(resized_frame, vial_bbox, liquid_predictor, solid_predictor, scale=scale, context=context)

            uncrop_ims, crop_ims = eval_yolo_batch(batch, vial_bbox, liquid_predictor, scale=scale)
            batch = []
            for uncrop_im, crop_im in zip(uncrop_ims, crop_ims):
                writer1.append_data(uncrop_im)
                writer2.append_data(crop_im)
            # if len(buff)>max_buff:
            #     buff.pop(0)
            pbar.update(1)
        else:
            cap.release()
            break
    writer1.close()
    writer2.close()
    pbar.close()
    LOG.info(f'Videos saved at ./output/insseg/')

    return

if __name__ == '__main__':
    """
    main func
    """
    segment_video()
