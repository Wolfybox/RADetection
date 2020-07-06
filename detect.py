import argparse
import time

from I3D.extract_flow_features import *
from I3D.extract_rgb_features import *
from AnoDetector.RN_Vanilla import RNVanilla


def min_max_normalize(scores):
    norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    return norm_scores


def predict_anomaly_scores(vfeats, net):
    # Initialize Model
    features = torch.nn.functional.normalize(vfeats, p=2, dim=1)
    feat_num = features.size()[0]
    # Prepare Segments
    segments = []
    if feat_num < 33:
        indices = torch.from_numpy(np.linspace(0, feat_num, 32, endpoint=False, dtype=np.int))
        for index in indices:
            segments.append(features[index].unsqueeze(0))
    else:
        indices = torch.from_numpy(np.linspace(0, feat_num, 33, endpoint=True, dtype=np.int))
        for k in range(indices.shape[0] - 1):
            cur_seg_feats = features[indices[k]:indices[k + 1]].mean(dim=0).unsqueeze(0)
            segments.append(cur_seg_feats)
    segments = torch.stack(segments)
    with torch.no_grad():
        return net(segments.cuda()).squeeze().cpu().numpy()


def merge_score(score1, score2):
    cs1 = score1.squeeze()
    cs2 = score2.squeeze()
    ms = cs1 * cs2
    return ms


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Road Accident Detection')
    parser.add_argument('--gpu', default=0, type=int, help='GPU Device No')
    parser.add_argument('--vdir', default='AnoVideos', type=str, help='Paths of videos to be detected')
    parser.add_argument('--outdir', default='Scores', type=str, help='Output Scores Directory')
    parser.add_argument('--rgb_mdir', default='AnoDetector/Vanilla_RGB.pt', type=str,
                        help='Path of the Pre-trained Model on RGB Features')
    parser.add_argument('--flow_mdir', default='AnoDetector/Vanilla_FLOW.pt', type=str,
                        help='Path of the Pre-trained Model on FLOW Features')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    video_dir = args.vdir
    rgb_net = RNVanilla(pretrained=True, model_dir=args.rgb_mdir, input_dim=1024).eval().cuda()
    flow_net = RNVanilla(pretrained=True, model_dir=args.flow_mdir, input_dim=1024).eval().cuda()
    fps_list = []
    for vname in tqdm(os.listdir(video_dir)):
        print(f'Detecting {vname}...')
        start_t = time.time()
        v_dir = os.path.join(video_dir, vname)
        print(f'Extracting RGB Features for {vname}...')
        rgb_feat, frame_num = extract_rgb_features(v_dir)
        print(f'Extracting FLOW Features for {vname}...')
        flow_feat, _ = extract_flow_features(v_dir)
        print(f'Detecting with RGB features on {vname}...')
        rgb_scores = predict_anomaly_scores(rgb_feat, net=rgb_net)
        rgb_scores = min_max_normalize(rgb_scores)
        print(f'Detecting with FLOW features on {vname}...')
        flow_scores = predict_anomaly_scores(flow_feat, net=flow_net)
        flow_scores = min_max_normalize(flow_scores)
        print(f'Merging Scores for {vname}...')
        joint_scores = merge_score(rgb_scores, flow_scores)
        print(f'Saving Scores for {vname}...')
        np.save(os.path.join(args.outdir, '{}.npy'.format(vname.split('.')[0])), joint_scores)
        end_t = time.time()
        print(f'fps:{frame_num // (end_t - start_t)}')
        fps_list.append(frame_num / (end_t - start_t))
    print('avg fps:{}'.format(sum(fps_list) // len(fps_list)))
