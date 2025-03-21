#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os
import sys
import os.path as osp
import torch
from Config import Config as cfg

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.utils.events import LOGGER
from yolov6.core.inferer import Inferer


class Detection:
    def __init__(self):
        self.weights=cfg.MODEL_WEIGHT
        # self.source=source
        self.webcam=False
        self.webcam_addr="0"
        self.yaml=cfg.MODEL_CLASS_YML
        self.img_size=640
        self.conf_thres=cfg.CONF_THRESH
        self.iou_thres=cfg.IOU_THRESH
        self.max_det=1000
        self.device='0'
        self.save_txt=False 
        self.not_save_img=False
        self.save_dir=None
        self.view_img=cfg.SHOW_IMAGE
        self.classes=None
        self.agnostic_nms=False
        self.project=osp.join(ROOT, 'runs/inference')
        self.name='exp'
        self.hide_labels=False
        self.hide_conf=True
        self.half=True
        self.ROI = cfg.MODEL_ROI
        
    def get_pipeline_details_and_run(self):
       
        if len(cfg.RTSP_URL) > 0:
            self.run(cfg.RTSP_URL)
        # else:
        #     self.run(self.custom_config.get('rtsp_url'))   
            
    def run(self,source):
        if str(source).__contains__('rtsp'):       
            self.webcam=True
            self.webcam_addr=source
        else:
            self.webcam=False        
        # create save dir
        if self.save_dir is None:
            self.save_dir = osp.join(self.project, self.name)
            save_txt_path = osp.join(self.save_dir, 'labels')
        else:
            save_txt_path = self.save_dir
        if (not self.not_save_img or self.save_txt) and not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            LOGGER.warning('Save directory already existed')
        if self.save_txt:
            save_txt_path = osp.join(self.save_dir, 'labels')
            if not osp.exists(save_txt_path):
                os.makedirs(save_txt_path)
        inferer = Inferer(source, self.webcam, self.webcam_addr, self.weights, self.device, self.yaml, self.img_size, self.half)
        inferer.infer(self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, self.max_det, self.save_dir, self.save_txt, not self.not_save_img, self.hide_labels, self.hide_conf, self.view_img)   
    

        if self.save_txt or not self.not_save_img:
            LOGGER.info(f"Results saved to {self.save_dir}")
            
            
if __name__ == "__main__":
    I = Detection()
    I.get_pipeline_details_and_run()
    