import os
os.environ["OPENCV_LOG_LEVEL"]="SILENT"
import os.path as ops
import argparse
import keyboard
import imageio
import cv2
import numpy as np
import tqdm
import torch
import device
import time
from local_utils.log_util import init_logger
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
import matplotlib.style as mplstyle
import matplotlib as mpl
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 1.0
mplstyle.use('fast')

classes = ["Homogeneous Reaction", "Heterogeneous Reaction", "Residue", "Empty", "Solid", "StirBar"]
colors = [(189/255.0, 16/255.0, 224/255.0), (245/255.0, 166/255.0, 35/255.0), (110/255.0, 226/255.0, 105/255.0), (248/255.0, 231/255.0, 28/255.0), (0/255.0, 60/255.0, 255/255.0), (60/255.0, 60/255.0, 60/255.0)]


solid_classes = ["Residue", "Solid", "StirBar"]
solid_colors = [(248/255.0, 231/255.0, 28/255.0), (0/255.0, 60/255.0, 255/255.0), (110/255.0, 226/255.0, 105/255.0)]
liquid_classes = ["Homo", "Hetero", "Empty", "Cap"]
liquid_colors = [(189, 16, 224), (245, 166, 35), (120, 120, 120), (60, 60, 60)]
LOG = init_logger.get_logger('instance_seg.log')
VIAL_SIZE = 1.8  # mL


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_path', type=str, default='')
    parser.add_argument('--insseg_cfg_path', type=str, default='./config/insseg.yaml')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--nms_iou', type=float, default=0.2)
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--create_plots', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./output/insseg')
    parser.add_argument('--use_text_prefix', action='store_true')
    parser.add_argument('--use_cameras', type=int, default=0)
    return parser.parse_args()


def initialize_vial_detector():
    model = torch.hub.load('./yolov5', 'custom', path='./vial/best.pt', source='local')
    return model


def initialize_yolo(conf=0.5, nms_iou=0.2):
    liquid_model = torch.hub.load('./yolov5', 'custom', path='./liquid/best.pt', source='local')
    # liquid_model.to('cuda')
    LOG.info(f"IOU threshold for NMS set to {nms_iou}")
    LOG.info(f"Confidence score threshold set to {conf}")
    liquid_model.conf = conf
    liquid_model.iou = nms_iou
    liquid_model.agnostic = True
    solid_model = torch.hub.load('./yolov5', 'custom', path='./solid/best.pt', source='local')
    return liquid_model, solid_model


def eval_yolo_batch(ims, boxes, liquid_predictor, scale=1.0, batch_size=32):
    volumes = np.zeros((len(ims), len(boxes), 5))
    segs = np.zeros((len(ims), len(boxes), 5))
    turbidity = np.zeros((len(ims), len(boxes), 500))
    colors = np.zeros((len(ims), len(boxes), 5, 3))
    batch = []
    ret_cropped = []
    ret_uncropped = []
    # start = time.time()
    for im_idx, im in enumerate(ims):
        for b_idx, box in enumerate(boxes):
            x, y, w, h, = box
            x, y, w, h = int(x * scale), int(y * scale), int(w * scale), int(h * scale)
            seg = im[y:y + h, x:x + w]
            turb = cv2.resize(seg.copy(), (100, 500))
            hsv = cv2.cvtColor(turb, cv2.COLOR_RGB2HSV)
            v = np.mean(hsv[:, :, -1], axis=-1)
            # LOG.info(v.size)
            turbidity[im_idx, b_idx] += v
            # seg = cv2.resize(seg, (640, 640))
            batch.append(seg.copy())
    # LOG.info(turbidity)
    pad = batch_size - len(batch)
    for _ in range(pad):
        batch.append(batch[-1].copy())
    # batch = torch.tensor(np.stack(batch))
    # batch = torch.permute(batch, (0, 3, 1, 2))
    # print(len(batch))
    # end = time.time()
    # LOG.info(f"Time to initialize batch {end-start}")
    # LOG.info(f"Batch length {len(batch)}")
    # start = time.time()
    results = liquid_predictor(batch, size=640)
    # return ims, ims, turbidity, colors, volumes, segs
    # end = time.time()
    # LOG.info(f"Inference time for {len(batch)}: {end-start}")
    # return ims, ims, turbidity, colors, volumes, segs
    # start = time.time()
    for im_idx, im in enumerate(ims):
        for box_idx, box in enumerate(boxes):
            x, y, w, h, = box
            x, y, w, h = int(x * scale), int(y * scale), int(w * scale), int(h * scale)
            lowest_h = 2
            bx_volumes = []
            bx_colors = []
            for boxp in results.xyxyn[im_idx * len(boxes) + box_idx].to('cpu'):
                list_box = boxp.tolist()
                classp = int(list_box[5])
                scorep = list_box[4]
                # if scorep < 0.25:
                #     continue
                if list_box[1] < lowest_h:
                    lowest_h = list_box[1]
                m_y = (list_box[1] + list_box[3]) / 2.0
                # if liquid_classes[classp] != "Empty":
                bx_volumes.append((list_box[3] - list_box[1], m_y, list_box[1]))
                boxp = boxp[:4] * torch.Tensor([w, h, w, h]).to('cpu') + torch.Tensor([x, y, x, y]).to('cpu')
                boxp = boxp.to('cpu')
                list_boxp = boxp.tolist()
                seg = im[int(list_boxp[1]):int(list_boxp[3]), int(list_boxp[0]):int(list_boxp[2])]
                # if liquid_classes[classp] != "Empty":
                bx_colors.append(((np.mean(seg, (0, 1))).astype(np.uint8), m_y))
                im = cv2.rectangle(im, tuple(boxp[:2].numpy().astype(int)), tuple(boxp[2:].numpy().astype(int)),
                                   liquid_colors[classp], 2)
                im = cv2.putText(im, f'{liquid_classes[classp]}, {scorep:.2f}', tuple(boxp[:2].numpy().astype(int)),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, liquid_colors[classp], 2)
            bx_volumes = sorted(bx_volumes, key=lambda x: x[1])
            bx_volumes = flip_volumes(bx_volumes)
            bx_colors = sorted(bx_colors, key=lambda x: x[1])
            bx_colors = flip_volumes(bx_colors)
            for idx, vol in enumerate(bx_volumes):
                volumes[im_idx, box_idx, idx] += vol[0] * VIAL_SIZE / (1 - lowest_h)  # height in volume graph
                segs[im_idx, box_idx, idx] += (1 - vol[2])
            for idx, col in enumerate(bx_colors):
                colors[im_idx, box_idx, idx] += col[0]
        # end = time.time()
        # LOG.info(f"Time to create visualizations {end-start}")
        # start = time.time()
        # im = cv2.resize(im, (1920, 1080))
        ret_cropped.append(im)
        ret_uncropped.append(im)
    # end = time.time()
    # LOG.info(f"Time to return inference {end-start}")
    # LOG.info(colors)
    # LOG.info(turbidity)
    # LOG.info(volumes)
    return ret_cropped, ret_uncropped, turbidity.astype(np.uint8), colors.astype(np.uint8), volumes.astype(
        np.float16), segs.astype(np.float16)


def get_vials(frame, vial_detector, VESSEL_THRESH):
    H, W, _ = frame.shape
    results = vial_detector(frame)
    results.save(save_dir='./output/insseg', exist_ok=True)  # saves vials images
    # LOG.info(results.xyxyn[0][:, 4])
    mask = results.xyxyn[0][:, 4] >= VESSEL_THRESH
    # LOG.info(mask)
    indices = torch.nonzero(mask)
    # LOG.info(indices)
    results.xyxyn[0] = torch.squeeze(results.xyxyn[0][indices], dim=1)
    # LOG.info(results.xyxyn[0][:, :4].shape)
    vial_bbox = results.xyxyn[0][:, :4].to('cpu') * torch.tensor([W, H, W, H]).to('cpu')
    vial_bbox[:, 2] = vial_bbox[:, 2] - vial_bbox[:, 0]
    vial_bbox[:, 3] = vial_bbox[:, 3] - vial_bbox[:, 1]
    vial_bbox = vial_bbox.to(torch.int32).tolist()
    vial_bbox = sorted(vial_bbox, key=lambda x: x[0])
    return vial_bbox


def create_plots(turbs, vols, segs):
    turbs = turbs.astype(np.float16)
    turbs /= 255.0
    num_frames = turbs.shape[0]
    num_vials = turbs.shape[1]
    max_segs = vols.shape[2]
    cols = num_vials
    for i in range(int(math.sqrt(num_vials) + 1), 1, -1):
        if num_vials % i == 0:
            cols = i
            break
    rows = num_vials // cols

    if num_frames == 1:
        fig, axs = plt.subplots(rows, cols, figsize=(8, 12))
        fig.suptitle('Turbidity grid')
        x = np.flip(np.linspace(0, 1, 500))
        fr = 0
        for i in range(num_vials):
            lines = []
            y = turbs[fr, i]
            row = i // cols
            col = i % cols
            if rows == 1 and cols == 1:
                ax = axs
            elif rows == 1:
                ax = axs[col]
            elif cols == 1:
                ax = axs[row]
            else:
                ax = axs[row, col]
            lines.append(ax.plot(x, y)[0])
            ax.set_title(f'Vial {i + 1}')
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            for line in lines:
                xdata, ydata = line.get_xdata(), line.get_ydata()
                line.set_xdata(ydata)
                line.set_ydata(xdata)
            for j in range(max_segs):
                if segs[fr, i, j] != 0:
                    ax.axhline(y=segs[fr, i, j], color='red', linestyle='--')
        plt.savefig('./output/insseg/turbidites.jpg')
        plt.close()
        return

    fig, axs = plt.subplots(cols, rows, figsize=(12, 8))
    fig.suptitle('Volume grid')
    if num_frames % 30 == 0:
        x = np.linspace(0, num_frames // 30, num_frames // 30)
    else:
        x = np.linspace(0, num_frames // 30 + 1, num_frames // 30 + 1)
    for i in tqdm.tqdm(range(num_vials)):
        v_vols = vols[:, i, :]
        for j in range(max_segs):
            v_vol = v_vols[:, j]
            any_data = np.any(v_vol != 0)
            if any_data:
                row = i // cols
                col = i % cols
                if rows == 1 and cols == 1:
                    ax = axs
                elif rows == 1:
                    ax = axs[col]
                elif cols == 1:
                    ax = axs[row]
                else:
                    ax = axs[col, row]
                ax.plot(x, v_vol[::30])
                ax.set_title(f'Vial {i + 1}')
    plt.savefig('./output/insseg/volumes.jpg')
    plt.close()

    turbidity_vid_writer = imageio.get_writer(f"./output/insseg/turbidities.mp4", fps=30)
    fig, axs = plt.subplots(rows, cols, figsize=(8, 12))
    # fig.suptitle('Turbidity grid') 
    x = np.flip(np.linspace(0, 1, 500))
    for fr in tqdm.tqdm(range(0, num_frames, 10)):
        fig.suptitle(f"Turbidity graph {fr / 30.0}s")
        for i in range(num_vials):
            lines = []
            y = turbs[fr, i]
            row = i // cols
            col = i % cols
            if rows == 1 and cols == 1:
                ax = axs
            elif rows == 1:
                ax = axs[col]
            elif cols == 1:
                ax = axs[row]
            else:
                ax = axs[row, col]
            lines.append(ax.plot(x, y)[0])
            ax.set_title(f'Vial {i + 1}')
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            for line in lines:
                xdata, ydata = line.get_xdata(), line.get_ydata()
                line.set_xdata(ydata)
                line.set_ydata(xdata)
            for j in range(max_segs):
                if segs[fr, i, j] != 0:
                    ax.axhline(y=segs[fr, i, j], color='red', linestyle='--')
        canvas = FigureCanvas(fig)
        canvas.draw()
        image_array = np.array(canvas.renderer.buffer_rgba())
        turbidity_vid_writer.append_data(image_array)
        for i in range(num_vials):
            row = i // cols
            col = i % cols
            if rows == 1 and cols == 1:
                ax = axs
            elif rows == 1:
                ax = axs[col]
            elif cols == 1:
                ax = axs[row]
            else:
                ax = axs[row, col]
            ax.clear()
    plt.close()
    turbidity_vid_writer.close()


def save_data(turbs, cols, vols, fps):
    os.makedirs("./output/raw_data", exist_ok=True)
    num_vials = vols.shape[1]
    max_num_segments = vols.shape[2]
    volume_df = pd.DataFrame({})
    color_df = pd.DataFrame({})
    volume_df["Time"] = [i / fps / 60 for i in range(vols.shape[0])]
    color_df["Time"] = [i / fps / 60 for i in range(vols.shape[0])]
    for i in range(num_vials):
        for j in range(max_num_segments):
            if np.any(vols[:, i, j] != 0):
                volume_df[f"vial {i + 1} segment {j + 1}"] = vols[:, i, j]
            if np.any(cols[:, i, j, :] != 0):
                color_df[f"vial {i + 1} segment {j + 1}"] = np.split(cols[:, i, j, :], cols.shape[0], axis=0)
        np.save(f"./output/raw_data/turb_vial_{i + 1}.npy", turbs[:, i, :])
    volume_df.to_csv("./output/raw_data/volumes.csv")
    color_df.to_csv("./output/raw_data/colors.csv")


def find_cams(num_cams):
    ret_caps = []
    device_list = device.getDeviceList()
    for i in range(100):
        try:
            cap = cv2.VideoCapture(i)
            if not (cap is None or not cap.isOpened()):
                ret_caps.append(i)
                cap.release()
        except:
            continue
    if len(ret_caps) < num_cams:
        LOG.warning(f"Found {len(ret_caps)} input video streams, but --use_cameras expects {num_cams}")
    LOG.opt(raw=True).info("Cameras found below:\n")
    for idx, camera in enumerate(device_list):
        LOG.opt(raw=True).info(str(ret_caps[idx]) + ': ' + camera[0] + '\n')
    cameras_set = False
    while not cameras_set:
        ans = input(
            "Input the cameras (first value from device list) you would like to use as a comma separated string eg: 0,1\n")
        idxs = ans.split(',')
        try:
            idxs = [int(idx) for idx in idxs]
            num_err = False
            for idx in idxs:
                if idx not in ret_caps:
                    LOG.error(f"Received index not in options {idx}")
                    num_err = True
                    break
            if num_err:
                continue
            elif len(idxs) != num_cams:
                conf = input(
                    f"Received {len(idxs)} cameras to use but --use_cameras required {num_cams}.\n Continue with {idxs} (y/n)")
                if conf == "y" or conf == "Y":
                    ret_caps = idxs
                    LOG.info(f"Cameras set to {ret_caps}")
                    cameras_set = True
            else:
                ret_caps = idxs[:]
                LOG.info(f"Cameras set to {ret_caps}")
                cameras_set = True
        except:
            LOG.error("Received a non-number camera index")

    return ret_caps


def segment_video():
    args = init_args()
    input_image_path = args.input_image_path
    cam_idx = []
    if input_image_path == "" and args.use_cameras == 0:
        LOG.error('Usage: required one of --input_image_path or --use_cameras')
        return
    elif args.use_cameras == 0:
        input_image_name = ops.split(input_image_path)[1]
        if not ops.exists(input_image_path):
            LOG.error('input video path: {:s} not exists'.format(input_image_path))
            return
    elif input_image_path == "":
        input_image_name = "Cameras"
        cam_idx += find_cams(args.use_cameras)
        if cam_idx == []:
            LOG.error('Could not find any  cameras attached')
            return
    else:
        LOG.info("Recieved both --input_image_path and --use_cameras, proceding with --input_image_path")
        input_image_name = ops.split(input_image_path)[1]
        if not ops.exists(input_image_path):
            LOG.error('input video path: {:s} not exists'.format(input_image_path))
            return
    LOG.info(f"Initializing vessel detector")
    VESSEL_THRESH = 0.7
    vial_detector = initialize_vial_detector()
    liquid_predictor, solid_predictor = initialize_yolo(conf=args.conf, nms_iou=args.nms_iou)
    if input_image_path == "":
        LOG.opt(raw=True).info("Since --use_cameras was used, waiting for experiment setup to complete\n")
        while True:
            ans = input("Continue? (y/n)")
            if ans == "y" or ans == "Y":
                LOG.info("To finish camera detections gracefully, press and hold the esc key to stop the analysis!")
                break
    # create_plots((np.random.normal(size=(120, 5, 500)) * 255).astype(np.uint8),
    #              np.random.normal(size=(120, 5, 5)), np.random.normal(size=(120, 5, 5)))
    # return
    if ops.split(input_image_path)[1].endswith(("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")):
        LOG.info("Detected Image type input, switing to image analysis")
        im = cv2.imread(input_image_path)
        im = cv2.resize(im, (1920, 1080))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        vial_bbox = get_vials(im, vial_detector, VESSEL_THRESH)
        if len(vial_bbox) == 0:
            LOG.error("Found no vials in the image")
            return
        LOG.info(f"Detected {len(vial_bbox)} vials at: {vial_bbox}")
        uncrop_ims, crop_ims, turbidity, color, volume, seg = eval_yolo_batch([im], vial_bbox, liquid_predictor,
                                                                              scale=1.0, batch_size=len(vial_bbox))
        create_plots(turbidity, volume, seg)
        crop_ims = cv2.cvtColor(crop_ims[0], cv2.COLOR_BGR2RGB)
        cv2.imwrite("./output/insseg/output.jpg", crop_ims)
        LOG.info("Detections saved at ./output/insseg/output.jpg")
        return

    if input_image_path == "":
        caps = [cv2.VideoCapture(i) for i in cam_idx]
    else:
        caps = [cv2.VideoCapture(input_image_path)]
    vial_bbox = []
    caps_opened = True
    _ = [caps_opened and cap.isOpened() for cap in caps]
    if caps_opened:
        ret = True
        frames = []
        for cap in caps:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            LOG.info(f"Current capture rate is {fps} fps")
            c_ret, frame = cap.read()
            frames.append(frame)
            ret = ret and c_ret
        if not ret:
            LOG.error("Could not read from file or cameras")
            return
        frame = np.concatenate(frames, axis=0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vial_bbox = get_vials(frame, vial_detector, VESSEL_THRESH)
    assert vial_bbox != []

    LOG.info(f"Detected {len(vial_bbox)} vials at: {vial_bbox}")
    # init cluster
    # create_plots((np.random.normal(size=(120, len(vial_bbox), 500))*255).astype(np.uint8), np.random.normal(size=(120, len(vial_bbox), 5)), np.random.normal(size=(120, len(vial_bbox), 5)))
    # save_data(np.random.normal(size=(100, 6, 200)), np.random.normal(size=(100, 6, 10, 3)), np.random.normal(size=(100, 6, 10)))
    writer1 = imageio.get_writer(f"./output/insseg/uncrop_{input_image_name.split('.')[0]}.mp4", fps=30)
    writer2 = imageio.get_writer(f"./output/insseg/crop_{input_image_name.split('.')[0]}.mp4", fps=30)
    if input_image_name=="Cameras":
        writer3 = imageio.get_writer(f"./output/insseg/capture_{input_image_name.split('.')[0]}.mp4", fps=30)
    LOG.info(f'Analysing Video')
    pbar = tqdm.tqdm()
    batch = []
    batch_size = args.batch_size // (len(vial_bbox))
    LOG.info(f"Batch size: {batch_size}")
    frame_count = 0
    turbidities, colors, volumes, segs = [], [], [], []
    if args.create_plots:
        LOG.info(
            "[WARNING] Plotting is very slow currently, the collected data is exporter as csv. We suggest using those instead")
    while caps_opened:
        _ = [caps_opened and cap.isOpened() for cap in caps]
        ret = True
        frames = []
        for cap in caps:
            c_ret, frame = cap.read()
            frames.append(frame)
            ret = ret and c_ret
        # ret, frame = cap.read()
        if ret:
            frame = np.concatenate(frames, axis=0)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            scale = 1920.0 / 1920.0
            frame_count += 1
            # resized_frame = cv2.resize(frame, (1920, 1080))
            resized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if input_image_name=="Cameras":
                w_frame = cv2.resize(resized_frame, (1920, 1080))
                writer3.append_data(w_frame)
            batch.append(resized_frame)
            # if frame_count%30 == 0: 
            #     vial_bbox = get_vials(resized_frame, vial_detector, VESSEL_THRESH) # vial_redetection
            if frame_count % batch_size != 0:
                pbar.update(1)
                continue
            uncrop_ims, crop_ims, turbidity, color, volume, seg = eval_yolo_batch(batch, vial_bbox, liquid_predictor,
                                                                                  scale=scale,
                                                                                  batch_size=args.batch_size)
            turbidities.append(turbidity)
            colors.append(color)
            volumes.append(volume)
            segs.append(seg)
            batch = []
            for uncrop_im, crop_im in zip(uncrop_ims, crop_ims):
                writer1.append_data(uncrop_im)
                writer2.append_data(crop_im)
            # if len(buff)>max_buff:
            #     buff.pop(0)
            pbar.update(1)
        else:
            LOG.info("Video ended or camera didn't return a frame, ending gracefully")
            _ = [cap.release() for cap in caps]
            if batch:
                uncrop_ims, crop_ims, turbidity, color, volume, seg = eval_yolo_batch(batch, vial_bbox,
                                                                                      liquid_predictor, scale=1.0,
                                                                                      batch_size=len(batch))
                turbidities.append(turbidity)
                colors.append(color)
                volumes.append(volume)
                segs.append(seg)
                for uncrop_im, crop_im in zip(uncrop_ims, crop_ims):
                    writer1.append_data(uncrop_im)
                    writer2.append_data(crop_im)
            break
        if args.use_cameras != 0 and keyboard.is_pressed('esc'):
            LOG.info("esc pressed, ending gracefully")
            _ = [cap.release() for cap in caps]
            if batch:
                uncrop_ims, crop_ims, turbidity, color, volume, seg = eval_yolo_batch(batch, vial_bbox,
                                                                                      liquid_predictor, scale=1.0,
                                                                                      batch_size=len(batch))
                turbidities.append(turbidity)
                colors.append(color)
                volumes.append(volume)
                segs.append(seg)
                for uncrop_im, crop_im in zip(uncrop_ims, crop_ims):
                    writer1.append_data(uncrop_im)
                    writer2.append_data(crop_im)
            break
    writer1.close()
    writer2.close()
    if input_image_name=="Cameras":
        writer3.close()
    pbar.close()
    turbidities = np.concatenate(turbidities, axis=0)
    colors = np.concatenate(colors, axis=0)
    segs = np.concatenate(segs, axis=0)
    volumes = np.concatenate(volumes, axis=0)
    volumes = rolling_average(volumes)
    # LOG.info(turbidities.shape)
    # LOG.info(colors.shape)
    # LOG.info(volumes.shape)
    save_data(turbidities, colors, volumes, fps)
    if args.create_plots:
        create_plots(turbidities, volumes, segs)
    LOG.info(f'Videos saved at ./output/insseg/')

    return


def rolling_average(volumes, rolling_window: int = 5):
    """
    take average of the #rolling window previous frames and return the new array
    :param volumes: np array (#frames, #vials, #bbox)
    :param rolling_window: number of frames to average
    """
    new_array = np.zeros(volumes.shape)
    for i in range(volumes.shape[0]):
        index = i + 1
        num_to_average = min(index, rolling_window)
        new_array[i] = np.mean(volumes[-num_to_average+index:index], axis=0)
    return new_array


def flip_volumes(bx_volumes):
    """
    assume there will always be empty detected, flip the liquid volumes if there are more than 1 liquid detected
    """
    if len(bx_volumes) >= 2:  # Ensure at least 2 elements in the list
        bx_volumes[1], bx_volumes[-1] = bx_volumes[-1], bx_volumes[1]  # Swap the second and last elements
    return bx_volumes


if __name__ == '__main__':
    """
    main func
    """
    segment_video()
