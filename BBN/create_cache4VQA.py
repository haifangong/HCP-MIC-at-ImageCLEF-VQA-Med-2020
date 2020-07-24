import _pickle as cPickle
import os
import json


RAD_dir = '/home/duadua/MVQA/SMMC/data'
train_answers = json.load(open(RAD_dir + '/trainset.json'))

token_lst = []
train_token_num = []
for ans_entry in train_answers:
    token = ans_entry['answer']
    if token not in token_lst:
        token_lst.append(token)
        train_token_num.append(1)
    else:
        index = token_lst.index(token)
        train_token_num[index] += 1

train_dict = {}
for i in range(len(train_token_num)):
    train_dict[token_lst[i]] = train_token_num[i]

train_dict = sorted(train_dict.items(), key=lambda x: x[1], reverse=True)

ans2label = {'no': 0, 'yes': 1}
label2ans = ['no', 'yes']
cnt = 2
for i in range(len(train_dict)):
    key = train_dict[i][0]
    if key != 'no' and key != 'yes':
        ans2label[key] = cnt
        label2ans.append(key)
        cnt += 1


cache_root = '/home/duadua/code-for-haifan/BBN/cache4VQA-Med'
name = 'trainval'
if not os.path.exists(cache_root):
    os.makedirs(cache_root)

cache_file = os.path.join(cache_root, name+'_ans2label.pkl')
cPickle.dump(ans2label, open(cache_file, 'wb'))
cache_file = os.path.join(cache_root, name+'_label2ans.pkl')
cPickle.dump(label2ans, open(cache_file, 'wb'))


def compute_target(answers_dset, ans2label, name, cache_root='/home/duadua/code-for-haifan/BBN/cache4VQA-Med'):
    """Augment answers_dset with soft score as label

    ***answers_dset should be preprocessed***

    Write result into a cache file
    """
    target = []
    count = 0
    for ans_entry in answers_dset:
        answers = ans_entry['answer']
        labels = []
        scores = []
        if answers in ans2label:
            scores.append(1.)
            labels.append(ans2label[answers])

        target.append({
            'qid': ans_entry['qid'],
            'image_name': ans_entry['image_name'],
            'labels': labels,
            'scores': scores
        })
    cache_file = os.path.join(cache_root, name+'_target.pkl')
    cPickle.dump(target, open(cache_file, 'wb'))
    return target

cache_path = '/home/duadua/code-for-haifan/BBN/cache4VQA-Med/trainval_ans2label.pkl'
ans2label = cPickle.load(open(cache_path, 'rb'))

# RAD_dir = 'F:/Medical-VQA/VQA-Med-2020/code/method/data_CLEF'
train_answers = json.load(open(RAD_dir + '/trainset.json'))
compute_target(train_answers, ans2label, 'train')

val_answers = json.load(open(RAD_dir + '/valset.json'))
compute_target(val_answers, ans2label, 'val')
#
test_answers = json.load(open(RAD_dir + '/testset.json'))
compute_target(test_answers, ans2label, 'test')

# cache_path = 'F:/Medical-VQA/BBN/cache4VQA-Med/test_target.pkl'
# val_target = cPickle.load(open(cache_path, 'rb'))
# print(val_target)