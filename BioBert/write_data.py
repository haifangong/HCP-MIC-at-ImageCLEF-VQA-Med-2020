import json
import os

save_dir = './data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

lst = []
file = json.load(open('../BBN-BioBert-Inference/data_CLEF/testset.json'))
for entry in file:
    qid = entry['qid']
    question = entry['question']
    if question.startswith('what'):
        label = 2
    else:
        if 'abnormal' in question:
            label = 1
        elif 'wrong' in question:
            label = 1
        else:
            label = 0
    item = {'qid': qid, 'question': question, 'label': label}
    
    lst.append(item)

with open(os.path.join(save_dir, 'testset.json'), 'w') as f:
        json.dump(lst, f)
