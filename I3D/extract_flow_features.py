import math
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as tf
from tqdm import tqdm

from I3D.I3D_Network import InceptionI3d
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

video_tf = tf.Compose([
    tf.Resize(size=(224, 224), interpolation=Image.CUBIC)
])

flow_tf = tf.Compose([
    tf.ToTensor(),
    tf.Normalize(mean=[0.5], std=[0.5])
])

clip_len = 16

i3d_flow = InceptionI3d(in_channels=2,
                        name='i3d_flow',
                        model_dir='I3D/flow_imagenet.pt').eval().cuda()


def get_videos_paths(frames_dir, feats_dir):
    feat_names = []
    for split in os.listdir(feats_dir):
        split_dir = os.path.join(feats_dir, split)
        for classname in os.listdir(split_dir):
            classdir = os.path.join(split_dir, classname)
            feat_names += [vname for vname in os.listdir(classdir)]

    videopaths = []
    for split in os.listdir(frames_dir):
        split_dir = os.path.join(frames_dir, split)
        for classname in os.listdir(split_dir):
            classdir = os.path.join(split_dir, classname)
            for vname in os.listdir(classdir):
                if '{}_rgb.pt'.format(vname) not in feat_names:
                    videopaths.append(os.path.join(classdir, vname))
    return videopaths


def load_videos(videopath):
    capture = cv2.VideoCapture(videopath)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if 'Normal' in videopath:
        frame_count = 5408 if frame_count > 5408 else frame_count
    count = 0
    retaining = True
    frames = []
    while count < frame_count and retaining:
        retaining, frame = capture.read()
        if frame is not None:
            frame = video_tf(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            frames.append(frame)
        count += 1
    capture.release()
    return frames


def cal_optical_flows(resized_frames):
    frame_num = len(resized_frames)
    last_frame = resized_frames[frame_num - 1]
    resized_frames.append(last_frame)
    # dtvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    # deep_flow = cv2.optflow.createOptFlow_DeepFlow()
    dis_flow = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    flow_list = []
    for i in range(frame_num):
        prev = np.array(resized_frames[i].convert('L'))
        cur = np.array(resized_frames[i + 1].convert('L'))
        # flow = estimate(prev, cur)
        # flow = dtvl1.calc(prev, cur, None)
        # flow = deep_flow.calc(prev, cur, None)
        flow = dis_flow.calc(prev, cur, None)
        imgx = (flow[..., 0] / 255) * 2 - 1
        imgy = (flow[..., 1] / 255) * 2 - 1
        # imgx = flow[..., 0]
        # imgy = flow[..., 1]
        # img = torch.cat([imgx, imgy], dim=0).permute(1, 2, 0)
        img = np.asarray([imgx, imgy])
        flow_list.append(img)
    return torch.tensor(flow_list)


def extract_flow_features(videopath):
    frames = load_videos(videopath)
    frame_num = len(frames)
    clip_num = math.ceil(frame_num / clip_len)
    total_frame = clip_len * clip_num
    remain = total_frame - frame_num
    frames += [frames[frame_num - 1]] * remain
    feat_num = math.floor(len(frames) / clip_len)
    flows = cal_optical_flows(frames)
    flow_feat_list = []
    for i in range(feat_num):
        start = clip_len * i
        end = clip_len * (i + 1)
        flow_clip = torch.stack([flows[k] for k in range(start, end)]).unsqueeze(0).permute(0, 2, 1, 3, 4).cuda()
        flow_feat = i3d_flow.extract_features(flow_clip).view(1024)
        flow_feat_list.append(flow_feat)
    flow_feat_tensor = torch.stack(flow_feat_list).cpu()
    return flow_feat_tensor, frame_num
