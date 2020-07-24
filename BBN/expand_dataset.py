import numpy as np
import scipy.stats


def train2dict(dtype='train', ytype='1920'):
    name_dict = {}
    with open('data/' + dtype + ytype + '/All_QA_Pairs_' + dtype + '.txt', 'r') as o:
        lines = o.readlines()

    for line in lines:
        name, q, a = line.strip().split('|')
        if a.lower() == 'yes' or a.lower() == 'no':
            continue
        elif a not in name_dict.keys():
            name_dict[a] = 1
        else:
            name_dict[a] += 1
    name_dict = sorted(name_dict.items(), key=lambda x: x[1], reverse=True)
    new_name_dict = {}
    for i in name_dict:
        new_name_dict[i[0]] = i[1]
    return new_name_dict


def val2dict(dtype, ytype, target_dict):
    new_dict = {}
    for i in target_dict:
        new_dict[i] = 0
    with open('data/' + dtype + ytype + '/All_QA_Pairs_' + dtype + '.txt', 'r') as o:
        lines = o.readlines()
    for line in lines:
        name, q, a = line.strip().split('|')
        if a in new_dict.keys():
            new_dict[a] += 1
    return new_dict


def renew_traindict(train_dict, item):
    if item.lower() == 'yes' or item.lower() == 'no':
        return train_dict
    else:
        train_dict[item] += 1
        return train_dict


def to_list(dict):
    v_list = []
    for i in dict:
        v_list.append(dict[i])
    return v_list


def KL_divergence(p, q):
    return scipy.stats.entropy(p, q)


def find_best_distribution(candidate_path):
    train_dict = train2dict()
    train_list = to_list(train_dict)
    q = np.asarray(train_list)

    val_dict = val2dict(dtype='val', ytype='2020', target_dict=train_dict)
    target_list = to_list(val_dict)
    p = np.asarray(target_list)
    score = KL_divergence(p, q)
    print(score)

    with open(candidate_path, 'r') as o:
        lines = o.readlines()

    for line in lines:
        name, q, a = line.strip().split('|')
        train = renew_traindict(train_dict, a)
        train_list = to_list(train)
        q = np.asarray(train_list)
        new_score = KL_divergence(p, q)
        if new_score < score:
            print(new_score)
            with open('selected.txt', 'a+') as f:
                f.write(line)


def get_score():
    train_dict = train2dict()
    train_list = to_list(train_dict)
    q = np.asarray(train_list)

    val_dict = val2dict(dtype='val', ytype='2020', target_dict=train_dict)
    target_list = to_list(val_dict)
    p = np.asarray(target_list)
    score = KL_divergence(p, q)
    print(score)

get_score()
# find_best_distribution('data/201942020/2019used_QA_Pairs.txt')
