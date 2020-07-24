import socket
import argparse
from datetime import datetime
import time
import os
import glob
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from question_classifier import Question_Classifier
from dataset import Question_Dataset, Dictionary
import _pickle as pickle
import json
import utils
from tensorboardX import SummaryWriter


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')

    # Model setting
    parser.add_argument('-bert_mode', type=str, default='base', choices=['base', 'l', 'xl', 'xxl'])
    parser.add_argument('-bert_pretrain', type=str, default='../BBN-BioBert-Inference/pretrain/biobert_v1.1_pubmed')
    parser.add_argument('-num_classes', type=int, default=3)
    parser.add_argument('-dictionary_path', type=str, default='../BBN-BioBert-Inference/data_CLEF/dictionary.pkl')
    parser.add_argument('-data_root', type=str, default='./data')

    # Training setting
    parser.add_argument('-seed', type=int, default=1234)
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-nepochs', type=int, default=200)
    parser.add_argument('-resume_epoch', type=int, default=0)
    parser.add_argument('-train_fold', type=str, default='BioBert')
    parser.add_argument('-run_id', type=int, default=-1)

    # Optimizer setting
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-weight_decay', type=float, default=5e-4)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-update_lr_every', type=int, default=50)

    parser.add_argument('-save_every', type=int, default=10)
    parser.add_argument('-log_every', type=int, default=2)

    # Testing setting
    parser.add_argument('-load_path', type=str, default='./run/BioBert/run_1/models/question_classifier_epoch-39.pth')

    return parser.parse_args()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    net = Question_Classifier(args.bert_mode, args.bert_pretrain, num_classes=3)
    net.load_state_dict(torch.load(args.load_path, map_location=lambda storage, loc: storage))

    torch.cuda.set_device(device=0)
    net.cuda()

    dictionary = Dictionary.load_from_file(args.dictionary_path)
    valset = Question_Dataset('val', dictionary, args.data_root, question_len=12)
    testset = Question_Dataset('test', dictionary, args.data_root, question_len=12)

    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    net.eval()
    val_acc = 0.0
    test_acc = 0.0

    with torch.no_grad():
        for ii, sample_batched in enumerate(valloader):
            question, label = sample_batched['question'], sample_batched['label']
            question, label = question.cuda(), label.cuda()

            out = net.forward(question)
            tmp_acc = utils.cal_acc(out, label)
            val_acc += (tmp_acc * question.shape[0])
        val_acc /= len(valset)

        for ii, sample_batched in enumerate(testloader):
            question, label = sample_batched['question'], sample_batched['label']
            question, label = question.cuda(), label.cuda()

            out = net.forward(question)
            tmp_acc = utils.cal_acc(out, label)
            test_acc += (tmp_acc * question.shape[0])
        test_acc /= len(testset)

        print('valset || questions: %d acc: %.4f' % (len(valset), val_acc))
        print('testset || questions: %d acc: %.4f' % (len(testset), test_acc))


if __name__ == '__main__':
    args = get_arguments()
    main(args)
