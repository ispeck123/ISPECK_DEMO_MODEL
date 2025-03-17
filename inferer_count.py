#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import cv2
import time
import math
import torch
import numpy as np
import os.path as osp

from tqdm import tqdm
from pathlib import Path
from PIL import ImageFont
from collections import deque

from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.data.datasets import LoadData
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.torch_utils import get_model_info
from harness_classifier import predict_and_return
from Config import Config as cfg
import uuid, datetime
from dbconn import insert_into_event_transaction
from yolov6.utils.other import is_inside_polygon

class Inferer:
    def __init__(self, source, ROI, webcam, webcam_addr, weights, device, yaml, img_size, half):
        
        self.__dict__.update(locals())

        # Init model
        self.device = device
        self.img_size = img_size
        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{device}' if cuda else 'cpu')
        self.model = DetectBackend(weights, device=self.device)
        self.stride = self.model.stride
        self.class_names = load_yaml(yaml)['names']
        self.img_size = self.check_img_size(self.img_size, s=self.stride)  # check image size
        self.half = half

        # Switch model to deploy status
        self.model_switch(self.model.model, self.img_size)

        # Half precision
        if self.half & (self.device.type != 'cpu'):
            self.model.model.half()
        else:
            self.model.model.float()
            self.half = False

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(next(self.model.model.parameters())))  # warmup

        # Load data
        self.webcam = webcam
        self.webcam_addr = webcam_addr
        self.files = LoadData(source, webcam, webcam_addr)
        self.source = source
        self.update_time = time.time()
        self.ROI=ROI


    def model_switch(self, model, img_size):
        ''' Model switch to deploy status '''
        from yolov6.layers.common import RepVGGBlock
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()

        LOGGER.info("Switch model to deploy modality.")

    def infer(self, conf_thres, iou_thres, classes, agnostic_nms, max_det, save_dir, save_txt, save_img, hide_labels, hide_conf, view_img=True, custom_config=None):
        ''' Model Inference and results visualization '''
        vid_path, vid_writer, windows = None, None, []
        fps_calculator = CalcFPS()
        allowed_class = ["Person"]
        ctr = 0
        update_interval_bool = True
        is_first_time = True
        for img_src, img_path, vid_cap in tqdm(self.files):
            ctr += 1
            if ctr==cfg.SKIP_FRAMES:
                ctr = 0
            else:
                continue
            img, img_src = self.precess_image(img_src, self.img_size, self.stride, self.half)
            img = img.to(self.device)
            if len(img.shape) == 3:
                img = img[None]
                # expand for batch dim
            t1 = time.time()
            pred_results = self.model(img)
            det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
            t2 = time.time()
            file_creation_date = datetime.datetime.now().date()
            if self.webcam:
                save_path = osp.join(save_dir, self.webcam_addr)
                txt_path = osp.join(save_dir, self.webcam_addr)
            else:
                # Create output files in nested dirs that mirrors the structure of the images' dirs
                rel_path = osp.relpath(osp.dirname(img_path), osp.dirname(self.source))
                save_path = osp.join(save_dir, rel_path, osp.basename(img_path))  # im.jpg
                txt_path = osp.join(save_dir, rel_path, osp.splitext(osp.basename(img_path))[0])
                os.makedirs(osp.join(save_dir, rel_path), exist_ok=True)

            gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            img_ori = img_src.copy()

            # check image and font
            assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
            self.font_check()
            c=0
            detected_list = []
            found_non_compliance = False
            if len(det):
                det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (self.box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    class_num = int(cls)  # integer class
                    
                    
                    
                    if self.class_names[class_num]=="person" and save_img==True:
                        c+=1
                        label = None if hide_labels else (self.class_names[class_num] if hide_conf else f'{self.class_names[class_num]} {conf:.2f}')
                        x1, y1, x2, y2 = torch.tensor(xyxy).tolist()
                        # print(x1, x2, y1, y2)
                        cropped_person = img_ori[int(y1):int(y2), int(x1):int(x2)]
                        new_label = predict_and_return(cropped_person)
                        if new_label not in detected_list:
                            detected_list.append(new_label)
                        # print(new_label)
                        # cv2.imshow("cropped_person", cropped_person)
                        if new_label is cfg.MODEL_ALERT_CLASS_LIST[0]:
                            x1,y1,x2,y2=torch.tensor(xyxy).tolist()
                            c_x, c_y = int((x1+x2)//2), int((y1+y2)//2)
                            person_p = (int(c_x),int(c_y))
                            color=(0,0,255)
                            if self.ROI:
                                if is_inside_polygon(self.ROI,person_p):
                                    # color=(0,0,255)
                                    found_non_compliance = True
                            else:
                                found_non_compliance = True
                        else:
                            color = (0,255,0)
                        
                        self.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, new_label, color=color)
                # self.people_count(img_ori,5)

                img_src = np.asarray(img_ori)

            # FPS counter
            fps_calculator.update(1.0 / (t2 - t1))
            avg_fps = fps_calculator.accumulate()
            
            if cfg.MODEL_ALERT_CLASS_LIST[0] in detected_list:
                if is_first_time:
                   is_first_time = False
                else:
                    update_interval_bool = self.countdown()
                    
            # print(found_non_compliance, update_interval_bool)
            if cfg.SAVE_IMAGE and found_non_compliance and update_interval_bool:
                folder = cfg.DIRECTORY_SAVE_IMAGE
                pipeline_id = custom_config.get('PipelineId')
                filename_uuid = str(uuid.uuid4())
                svg_file_name = f"{filename_uuid}.jpg"
                # create subfolder
                if not os.path.exists(f"{folder}/pipeline_{pipeline_id}/{file_creation_date}"):
                    os.makedirs(f"{folder}/pipeline_{pipeline_id}/{file_creation_date}")

                cv2.imwrite(f"{folder}/pipeline_{pipeline_id}/{file_creation_date}/{svg_file_name}", img_src,
                            [cv2.IMWRITE_JPEG_QUALITY, 40])

                insert_into_event_transaction({
                    'site_id': custom_config.get('SiteId'),
                    'area_id': custom_config.get('AreaId'),
                    'cam_id': custom_config.get('CameraId'),
                    'pipeline_id': custom_config.get('PipelineId'),
                    'class_id': custom_config.get('ClassId'),
                    'model_id': custom_config.get('ModelId'),
                    'event_type': cfg.MODEL_ALERT_CLASS_LIST[0],
                    'bbox': torch.tensor(xyxy).tolist(),
                    'score': conf,
                    'filename': svg_file_name
                })
                self.update_time = time.time()
                update_interval_bool = False
                
            if self.ROI:
                pts = np.array(self.ROI,np.int32)
                pts = pts.reshape((-1, 1, 2))
                img_src = cv2.polylines(img_src, [pts],True, (0,255,0), 2)

            if view_img:
                if img_path not in windows:
                    windows.append(img_path)
                    cv2.namedWindow(str(img_path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(img_path), img_src.shape[1], img_src.shape[0])
                cv2.imshow(str(img_path), img_src)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Save results (image with detections)
            if save_img:
                if self.files.type == 'image':
                    cv2.imwrite(save_path, img_src)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, img_ori.shape[1], img_ori.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(img_src)

    @staticmethod
    def precess_image(img_src, img_size, stride, half):
        '''Process image before image inference.'''
        image = letterbox(img_src, img_size, stride=stride)[0]
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size,list) else [new_size]*2

    def make_divisible(self, x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    @staticmethod
    def draw_text(
        img,
        text,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        pos=(0, 0),
        font_scale=1,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0),
    ):

        offset = (5, 5)
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        rec_start = tuple(x - y for x, y in zip(pos, offset))
        rec_end = tuple(x + y for x, y in zip((x + text_w, y + text_h), offset))
        cv2.rectangle(img, rec_start, rec_end, text_color_bg, -1)
        cv2.putText(
            img,
            text,
            (x, int(y + text_h + font_scale - 1)),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

        return text_size
    
    @staticmethod
    def people_count(image,count):
        h,w,_ = image.shape
        print(h,w)
        cv2.rectangle(image, (20,50), (100,80), (255,0,0), -1, cv2.LINE_AA)

    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)

    @staticmethod
    def font_check(font='./yolov6/utils/Arial.ttf', size=10):
        # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
        assert osp.exists(font), f'font path not exists: {font}'
        try:
            return ImageFont.truetype(str(font) if font.exists() else font.name, size)
        except Exception as e:  # download if missing
            return ImageFont.truetype(str(font), size)

    @staticmethod
    def box_convert(x):
        # Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    @staticmethod
    def generate_colors(i, bgr=False):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        palette = []
        for iter in hex:
            h = '#' + iter
            palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
        num = len(palette)
        color = palette[int(i) % num]
        return (color[2], color[1], color[0]) if bgr else color
    
    def countdown(self):
        diff = abs(self.update_time - time.time())
        # print(round(diff), "Time difeenc", cfg.INSIGHTS_UPDATE_MODE_TIME_INTERVAL)
        if round(diff) >= cfg.INSIGHTS_UPDATE_MODE_TIME_INTERVAL:
            return True
        else:
            return False

class CalcFPS:
    def __init__(self, nsamples: int = 50):
        self.framerate = deque(maxlen=nsamples)

    def update(self, duration: float):
        self.framerate.append(duration)

    def accumulate(self):
        if len(self.framerate) > 1:
            return np.average(self.framerate)
        else:
            return 0.0
