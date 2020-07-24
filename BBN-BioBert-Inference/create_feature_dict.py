import argparse
import torch
from torch.utils.data import DataLoader
from image_classifier import Network
from question_classifier import Question_Classifier
from tqdm import tqdm
import os
import json
from dataset import VQAFeatureDataset, Dictionary
from torchvision import models


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')

    # Model setting
    parser.add_argument('-bert_mode', type=str, default='base', choices=['base', 'l', 'xl', 'xxl'])
    parser.add_argument('-bert_pretrain', type=str, default='./pretrain/biobert_v1.1_pubmed')
    parser.add_argument('-ques_num_classes', type=int, default=3)
    parser.add_argument('-img_num_classes', type=int, default=330)
    parser.add_argument('-dictionary_path', type=str, default='./data/dictionary.pkl')
    parser.add_argument('-data_root', type=str, default='./data')

    # Testing setting
    parser.add_argument('-split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-backbone_type', type=str, default='bbn_ress50')
    parser.add_argument('-img_model_path', type=str,
                        default='/home/duadua/MVQA/code-for-haifan/BBN/output/VQAMed2020/RESX.BBN.VQAMed2020.res50.350epoch.1920/models/best_model.pth')
    parser.add_argument('-cls2_model_path', type=str,
                        default='/home/duadua/MVQA/code-for-haifan/BBN-BioBert-Inference/pretrain/class.pth')
    parser.add_argument('-ques_model_path', type=str,
                        default='../BioBert/run/BioBert/run_1/models/question_classifier_epoch-39.pth')

    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dictionary = Dictionary.load_from_file(args.dictionary_path)
    feature_dict = {}

    ques_net = Question_Classifier(args.bert_mode, args.bert_pretrain, num_classes=args.ques_num_classes)
    img_net = Network(backbone_type=args.backbone_type, num_classes=args.img_num_classes)
    cls_net = models.resnet34(pretrained=False, num_classes=2)
    cls_net = cls_net.cuda()
    ques_net = ques_net.cuda()
    img_net = img_net.cuda()

    cls_net.load_state_dict(torch.load(args.cls2_model_path))
    ques_net.load_state_dict(torch.load(args.ques_model_path, map_location=lambda storage, loc: storage))
    img_net.load_model(args.img_model_path)

    eval_dset = VQAFeatureDataset(args.split, dictionary, args.data_root, question_len=12, clip=True)
    eval_loader = DataLoader(eval_dset, args.batch_size, shuffle=False, num_workers=2)

    cls_net.eval()
    ques_net.eval()
    img_net.eval()

    gt_list = []
    with torch.no_grad():
        for v, q, a, ans_type, q_types, image_name in tqdm(iter(eval_loader)):
            v, q, a = v.cuda(), q.cuda(), a.cuda()
            v = v.reshape(v.shape[0], 3, 224, 224)
            q_prob = ques_net(q)  # ques_num_classes
            q_prob = q_prob[0]  # [0: closed-ended-normal, 1: closed-ended-abnormal 2: open-ended]
            q_type = torch.argmax(q_prob)

            v_prob, feature = img_net(v)  # 1 x img_num_classes

            if q_type == 0 or q_type == 1:
                continue
            else:
                feature = feature.cpu().numpy().tolist()
                temp_list = []
                for i in feature:
                    temp_list.append(round(i, 4))
                gt = torch.argmax(a[0]).item()
                if gt not in gt_list:
                    gt_list.append(gt)
                    feature_dict[gt] = [temp_list]
                elif gt in gt_list:
                    feature_dict[gt].append(temp_list)
        json.dump(feature_dict, open('feature_dict.json', 'w'))


if __name__ == '__main__':
    args = get_arguments()
    main(args)
