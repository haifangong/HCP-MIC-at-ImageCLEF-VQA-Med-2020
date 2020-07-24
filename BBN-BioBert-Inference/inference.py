import argparse
import torch
from torch.utils.data import DataLoader
from image_classifier import Network
from question_classifier import Question_Classifier
from tqdm import tqdm
import numpy as np
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
    parser.add_argument('-model_name', type=str, default='resnest-crop-retrieval')
    parser.add_argument('-split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-backbone_type', type=str, default='bbn_ress50')
    parser.add_argument('-img_model_path', type=str, default='./pretrain/specific_abnormality_classifier.pth')
    parser.add_argument('-cls2_model_path', type=str, default='./pretrain/normal_abnormal_classifier.pth')
    parser.add_argument('-ques_model_path', type=str, default='./pretrain/question_classifier.pth')
    parser.add_argument('-feature_dict_path', type=str, default='./pretrain/fd.json')
    parser.add_argument('-out_dir', type=str, default='./results')

    return parser.parse_args()


def main(args):
    softmax = torch.nn.Softmax(dim=-1)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dictionary = Dictionary.load_from_file(args.dictionary_path)

    ques_net = Question_Classifier(args.bert_mode, args.bert_pretrain, num_classes=args.ques_num_classes)
    img_net = Network(backbone_type=args.backbone_type, num_classes=args.img_num_classes)
    cls_net = models.resnet34(pretrained=False, num_classes=2)
    cls_net = cls_net.cuda()
    ques_net = ques_net.cuda()
    img_net = img_net.cuda()

    cls_net.load_state_dict(torch.load(args.cls2_model_path))
    ques_net.load_state_dict(torch.load(args.ques_model_path, map_location=lambda storage, loc: storage))
    img_net.load_model(args.img_model_path)
    fd = json.load(open(args.feature_dict_path, 'r'))

    eval_dset = VQAFeatureDataset(args.split, dictionary, args.data_root, question_len=12, clip=True)
    eval_loader = DataLoader(eval_dset, args.batch_size, shuffle=False, num_workers=2)

    label2ans = eval_dset.label2ans  # ans2label = {'no': 0, 'yes': 1, diseases...}

    cls_net.eval()
    ques_net.eval()
    img_net.eval()

    score = 0
    closed_ques_num = 28
    closed_score = 0
    cnt = 0

    out_dir = os.path.join(args.out_dir, args.split, args.model_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    pred_file = open(out_dir + '/prediction.txt', 'w')
    ques_pred_file = open(out_dir + '/question_type_prediction.txt', 'w')
    all_list = [0 for i in range(332)]
    true_list = [0 for i in range(332)]

    csc = 0
    csctotal = 0
    with torch.no_grad():
        for v, q, a, ans_type, q_types, image_name in tqdm(iter(eval_loader)):
            v, q, a = v.cuda(), q.cuda(), a.cuda()
            v = v.reshape(v.shape[0], 3, 224, 224)
            q_prob = ques_net(q)  # 1 x ques_num_classes
            q_prob = q_prob[0]  # [0: closed-ended-normal, 1: closed-ended-abnormal 2: open-ended]
            q_type = torch.argmax(q_prob)

            v_prob, feature = img_net(v)  # 1 x img_num_classes

            if q_type == 0:
                normal_prob2 = softmax(cls_net(v)[0])[0].item()
                abnormal_prob = 1 - normal_prob2
                pred = 0 if abnormal_prob > normal_prob2 else 1

            elif q_type == 1:
                normal_prob2 = softmax(cls_net(v)[0])[0].item()
                abnormal_prob = 1 - normal_prob2
                pred = 1 if abnormal_prob > normal_prob2 else 0
            else:
                disease_prob = softmax(v_prob)
                prob1, pred_idx = torch.topk(v_prob, 5, dim=-1)
                prob1 = softmax(prob1[0])

                if prob1[0] > 0.6:
                    csctotal += 1
                    pred = torch.argmax(disease_prob) + 2
                else:
                    pred_idx = pred_idx.cpu().numpy().tolist()[0]
                    p_idx = []
                    for i in pred_idx:
                        p_list = []
                        for fdict in fd[str(i + 2)]:
                            fdict = np.asarray(fdict)
                            fdict = torch.from_numpy(fdict)
                            cs = torch.cosine_similarity(feature.cpu(), fdict, dim=1)
                            p_list.append(cs.item())
                        p_idx.append(max(p_list))
                    pred = pred_idx[p_idx.index(max(p_idx))] + 2

            if args.split != 'test':
                gt = torch.argmax(a[0])
                all_list[gt.item()] += 1
                if pred == gt:
                    if pred > 1:
                        csc += 1
                    true_list[gt.item()] += 1
                    score += 1
                    if cnt < closed_ques_num:
                        closed_score += 1

            img_name = image_name[0]
            pred_ans = label2ans[pred]
            pred_file.write(img_name[:-4] + '|' + pred_ans + '\n')
            ques_pred_file.write(img_name[:-4] + '|' + str(q_type.cpu().numpy()) + '\n')
            cnt += 1

        if args.split != 'test':
            open_score = score - closed_score
            score = (score * 100.0) / cnt
            open_score = (open_score * 100.0) / (cnt - closed_ques_num)
            closed_score = (closed_score * 100.0) / closed_ques_num

            file = open(out_dir + '/score.txt', 'w')
            file.write('score: %.4f\n' % (score))
            file.write('closed score: %.4f\n' % (closed_score))
            file.write('open score: %.4f\n' % (open_score))
            print(csc / csctotal)
            print('score: %.4f' % (score))


if __name__ == '__main__':
    args = get_arguments()
    main(args)
