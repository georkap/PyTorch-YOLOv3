# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:08:11 2018

script to run Yolov3 on a video and save the video with overlaid detections and a text file

@author: GEO
"""
import os
import argparse
import time
import numpy as np

import torch
import cv2

from models import Darknet
from utils.utils import load_classes, non_max_suppression

def prepare_video_writer(vid_path, size, fps=30):
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    vid_dir = os.path.dirname(vid_path)
    if not os.path.exists(vid_dir):
        os.makedirs(vid_dir)
    writer = cv2.VideoWriter(vid_path, fourcc, fps, size)
    return writer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, help='path to video')
    parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
    parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
    parser.add_argument('--class_thresh', type=float, default=0.25)
    parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--resize_input', default=False, action='store_true')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
    opt = parser.parse_args()
    print(opt)
    
    cuda = torch.cuda.is_available() and opt.use_cuda
    # Set up model
    model = Darknet(opt.config_path, img_size=opt.img_size)
    model.load_weights(opt.weights_path)
    
    if cuda:
        model.cuda()
    
    model.eval() # Set in evaluation mode
    
    classes = load_classes(opt.class_path) # Extracts class labels from file
        
    print ('\nPerforming object detection:')
#    prev_time = time.time()
    
    cap = cv2.VideoCapture(opt.video_file)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_vid_path = os.path.join(r"outputs", 
                                   os.path.basename(opt.video_file).split('.')[0]+"_detections_{}.mp4".format(int(100*opt.class_thresh)))
    
    writer = prepare_video_writer(output_vid_path, (video_width, video_height), fps)
    print("Saving output video at {}".format(output_vid_path))
#    cap.set(cv2.CAP_PROP_POS_FRAMES, 550)

    resize_input=opt.resize_input
    # Get detections
    with torch.no_grad():
        
        frame_iter = 0
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if ret:
                # turn ret into input
                img = frame / 255.
                if resize_input:
                    img = cv2.resize(img, (opt.img_size, opt.img_size))
                    resize_ratio = (video_height/opt.img_size, video_width/opt.img_size)
                img = np.transpose(img, (2, 0, 1))
                # As pytorch tensor
                img = torch.from_numpy(img).float().cuda()
                img.unsqueeze_(0)
                detections = model(img)
                detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)
                # overlay detections on frame and write to video writer
                out_img = frame.copy()
#                out_img = cv2.resize(frame/255., (opt.img_size, opt.img_size)).copy()
                # Draw bounding boxes and labels of detections
                if detections[0] is not None:
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]: # get the first image of the batch, works because batch=1
                        if classes[int(cls_pred)] != 'person': # keep only person detections
                            continue
                        if cls_conf < opt.class_thresh:
                            continue
                        if resize_input:
                            y1 *= resize_ratio[0]
                            y2 *= resize_ratio[0]
                            x1 *= resize_ratio[1]
                            x2 *= resize_ratio[1]
                        box_h = y2-y1
                        box_w = x2-x1
    
                        cv2.rectangle(out_img, (x1, y1), (x1 + box_w, y1 + box_h), (0,0,255), 3)
                        cv2.putText(out_img, str(classes[int(cls_pred)]), (x1 + 10, y1 + box_h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255,255,255),1)
#                    cv2.imshow("1",out_img)
#                    cv2.waitKey(0)
                writer.write(out_img)
                # TODO: save detections to txt
                frame_iter += 1
                time_for_frame = time.time() - start_time
                print("\rCompleted:[{}/{}] with {:.3f} fps".format(frame_iter, num_frames, 1./time_for_frame),end="")
            else:
                cap.release()
                break
    try:
        writer.release()
    except:
        print("didn't release")


if __name__=='__main__':
    main()