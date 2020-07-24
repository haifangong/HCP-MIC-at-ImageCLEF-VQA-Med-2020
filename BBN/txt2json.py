import json
import pickle

from PIL import Image
import numpy
import _pickle as cPickle
import numpy as np
"""
This code should be run in the root dir.
"""


def text2json(dtype='train', ytype='2020'):
    diction = []
    id2idx = {}
    with open("data/" + dtype + ytype + '/All_QA_Pairs_' + dtype + '.txt', 'r') as o:
        lines = o.readlines()

    count = 0
    for line in lines:
        info = line.strip().split('|')
        img_name = info[0] + '.jpg'
        ans = info[2]
        ans_typ = 'CLOSED' if ans.lower() == 'no' or ans.lower() == 'yes' else 'OPEN'
        que_typ = 'ABN'

        id2idx[img_name] = count
        enum_dict = {'qid': count,
                     'image_name': img_name,
                     'question': info[1],
                     'question_type': que_typ,
                     'answer': info[2],
                     'answer_type': ans_typ,
                     }
        diction.append(enum_dict)
        count += 1

    with open("data/" + dtype + 'set.json', 'w') as f:
        json.dump(diction, f)

    with open("data/" + dtype + 'imgid2idx.json', 'w') as f:
        json.dump(id2idx, f)


if __name__ == "__main__":
    text2json(dtype='train', ytype='2020')
