import socket
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
    parser.add_argument('-batch_size', type=int, default=10)
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

    return parser.parse_args()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    net = Question_Classifier(args.bert_mode, args.bert_pretrain, num_classes=3)

    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    if args.resume_epoch != 0:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', args.train_fold, 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', args.train_fold, 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    if args.run_id >= 0:
        run_id = args.run_id

    save_dir = os.path.join(save_dir_root, 'run', args.train_fold, 'run_'+str(run_id))
    log_dir = os.path.join(save_dir, datetime.now().strftime('%b%d_%H-%M-%M%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    logger = open(os.path.join(save_dir, 'log.txt'), 'w')
    logger.write('optim: SGD \nlr=%.4f\nweight_decay=%.4f\nmomentum=%.4f\nupdate_lr_every=%d\nseed=%d\n' % 
        (args.lr, args.weight_decay, args.momentum, args.update_lr_every, args.seed))

    if not os.path.exists(os.path.join(save_dir, 'models')):
        os.makedirs(os.path.join(save_dir, 'models'))

    if args.resume_epoch == 0:
        print('Training from scratch...')
    else:
        net_resume_path = os.path.join(save_dir, 'models', 'mcnet_epoch-' + str(args.resume_epoch - 1) + '.pth')
        print('Initializing weights from: {}, epoch: {}...'.format(save_dir, resume_epoch))
        net.load_state_dict(torch.load(net_resume_path, map_location=lambda storage, loc: storage))

    torch.cuda.set_device(device=0)
    net.cuda()

    net_optim = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    dictionary = Dictionary.load_from_file(args.dictionary_path)
    trainset0 = Question_Dataset('train0', dictionary, args.data_root, question_len=12)
    trainset1 = Question_Dataset('train1', dictionary, args.data_root, question_len=12)
    trainset2 = Question_Dataset('train2', dictionary, args.data_root, question_len=12)
    valset = Question_Dataset('val', dictionary, args.data_root, question_len=12)
    testset = Question_Dataset('test', dictionary, args.data_root, question_len=12)

    trainloader0 = DataLoader(trainset0, batch_size=args.batch_size, shuffle=True, num_workers=2)
    trainloader1 = DataLoader(trainset1, batch_size=args.batch_size, shuffle=True, num_workers=2)
    trainloader2 = DataLoader(trainset2, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_iter_tr = len(trainloader0)
    nitrs = args.resume_epoch * num_iter_tr
    nsamples = args.batch_size * nitrs
    print('each_epoch_num_iter: %d' % (num_iter_tr))

    global_step = 0

    epoch_losses = []
    recent_losses = []
    start_t = time.time()
    print('Training Network')

    for epoch in range(args.resume_epoch, args.nepochs):
        
        net.train()

        epoch_losses = []

        for ii, (sample_batched0, sample_batched1, sample_batched2) in enumerate(zip(trainloader0, trainloader1, trainloader2)):
            question0, label0 = sample_batched0['question'], sample_batched0['label']
            question0, label0 = question0.cuda(), label0.cuda()
            question1, label1 = sample_batched1['question'], sample_batched1['label']
            question1, label1 = question1.cuda(), label1.cuda()
            question2, label2 = sample_batched2['question'], sample_batched2['label']
            question2, label2 = question2.cuda(), label2.cuda()

            global_step += args.batch_size

            out0 = net.forward(question0)
            out1 = net.forward(question1)
            out2 = net.forward(question2)

            loss0 = utils.CELoss(logit=out0, target=label0, reduction='mean')
            loss1 = utils.CELoss(logit=out1, target=label1, reduction='mean')
            loss2 = utils.CELoss(logit=out2, target=label2, reduction='mean')
            loss = (loss0 + loss1 + loss2) / 3
            
            trainloss = loss.item()
            epoch_losses.append(trainloss)
            if len(recent_losses) < args.log_every:
                recent_losses.append(trainloss)
            else:
                recent_losses[nitrs % len(recent_losses)] = trainloss
            net_optim.zero_grad()
            loss.backward()
            net_optim.step()
          
            nitrs += 1
            nsamples += args.batch_size

            if nitrs % args.log_every == 0:
                meanloss = sum(recent_losses) / len(recent_losses)
                
                print('epoch: %d ii: %d trainloss: %.2f timecost:%.2f secs'%(
                    epoch, ii, meanloss, time.time()-start_t))
                writer.add_scalar('data/trainloss',meanloss,nsamples)                     

        # validation
        net.eval()
        val_acc = 0.0
        test_acc = 0.0

        for ii, sample_batched in enumerate(valloader):
            question, label = sample_batched['question'], sample_batched['label']
            question, label = question.cuda(), label.cuda()

            out = net.forward(question)
            tmp_acc = utils.cal_acc(out, label)
            val_acc += (tmp_acc * question.shape[0])
        val_acc /= len(valset)

        for ii, sample_batched in enumerate(valloader):
            question, label = sample_batched['question'], sample_batched['label']
            question, label = question.cuda(), label.cuda()

            out = net.forward(question)
            tmp_acc = utils.cal_acc(out, label)
            test_acc += (tmp_acc * question.shape[0])
        test_acc /= len(testset)

        print('Validation:')
        print('epoch: %d, val_questions: %d val_acc: %.4f' % (epoch, len(valset), val_acc))
        print('epoch: %d, test_questions: %d test_acc: %.4f' % (epoch, len(testset), test_acc))
        writer.add_scalar('data/valid_acc', val_acc, nsamples)

        if epoch % args.save_every == args.save_every - 1:
            net_save_path = os.path.join(save_dir, 'models', 'question_classifier_epoch-' + str(epoch) + '.pth')
            torch.save(net.state_dict(), net_save_path)
            print("Save net at {}\n".format(net_save_path))

        if epoch % args.update_lr_every == args.update_lr_every - 1:
            lr_ = utils.lr_poly(args.lr, epoch, args.nepochs, 0.9)
            print('(poly lr policy) learning rate: ', lr_)
            net_optim = optim.SGD(
                net.parameters(),
                lr=lr_,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )


if __name__ == '__main__':
    args = get_arguments()
    main(args)
