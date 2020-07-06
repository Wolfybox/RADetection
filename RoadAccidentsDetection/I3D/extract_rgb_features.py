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

rgb_tf = tf.Compose([
    tf.ToTensor(),
    tf.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

clip_len = 16

i3d_rgb = InceptionI3d(in_channels=3,
                       name='i3d_rgb',
                       model_dir='I3D/rgb_imagenet.pt').eval().cuda()


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


def cal_rgb_frames(resized_frames):
    return [rgb_tf(f) for f in resized_frames]


def extract_rgb_features(videopath):
    frames = load_videos(videopath)
    frame_num = len(frames)
    clip_num = math.ceil(frame_num / clip_len)
    total_frame = clip_len * clip_num
    remain = total_frame - frame_num
    frames += [frames[frame_num - 1]] * remain
    rgbs = cal_rgb_frames(frames)
    feat_num = math.floor(len(rgbs) / clip_len)
    rgb_feat_list = []
    for i in range(feat_num):
        start = clip_len * i
        end = clip_len * (i + 1)
        rgb_clip = torch.stack([rgbs[j] for j in range(start, end)]).unsqueeze(0).permute(0, 2, 1, 3, 4).cuda()
        rgb_feat = i3d_rgb.extract_features(rgb_clip).view(1024)
        rgb_feat_list.append(rgb_feat)
    rgb_feat_tensor = torch.stack(rgb_feat_list).cpu()
    return rgb_feat_tensor, frame_num


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    feats_dir = '/home/yangzehua/UCF_Crimes/CADP_RGB_Features'

    # with open('/home/yangzehua/RoadAccidentsDetector/ucf_train_test_info/Train_Clean.txt', 'r') as f:
    #     train_videos = f.readlines()
    #     train_videos = [line.strip() for line in train_videos]
    # with open('/home/yangzehua/RoadAccidentsDetector/ucf_train_test_info/Test_Clean.txt', 'r') as f:
    #     test_videos = f.readlines()
    #     test_videos = [line.strip() for line in test_videos]
    # videolist = train_videos + test_videos

    with open('/home/yangzehua/RoadAccidentsDetector/ucf_train_test_info/CADP_Test.txt', 'r') as f:
        test_videos = f.readlines()
        test_videos = [line.strip() for line in test_videos]
    videolist = test_videos

    finished = []
    for vp in tqdm(videolist):
        try:
            print('Extracting RGB for {}'.format(vp))
            extract_rgb_features(vp)
            finished.append(vp)
        except Exception:
            with open('log/rgb_cadp.txt', 'w') as f:
                for fin in finished:
                    f.write('{}\n'.format(fin))
