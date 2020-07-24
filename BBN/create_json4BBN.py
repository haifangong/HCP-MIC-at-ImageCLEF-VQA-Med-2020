import PIL.Image as Image
import os
import json
import _pickle as cPickle

# for classification: 330 diseases
# for VQA: 332 candidate answers (normal + unclear_abnormal + 330 dieases)
# Do remember to modify the cache path, image dir, json path, output_name and closed_answer_num

json_path = '/home/duadua/MVQA/SMMC/data/valset.json'  # txt2json.py
cache_path = '/home/duadua/MVQA/HPC-MIC at ImageCLEF VQA-Med 2020/BBN/cache4VQA-Med'  # create_cache4VQA.py
img_dir = '/home/duadua/MVQA/SMMC/data/val2020/images/'
output_name = './jsons/VQA_Med_1920_val.json'
closed_answer_num = 28


ans2label = cPickle.load(open(cache_path, 'rb'))
cnt = 0
valid_id = 0
data_dict = {'annotations': [], 'num_classes': 330}
file = json.load(open(json_path))

for item in file:
    img_name, ans = item['image_name'][:-4], item['answer']
    sample_dict = {}

    if cnt > closed_answer_num-1:
        sample_dict['image_id'] = valid_id
        sample_dict['fpath'] = img_dir + img_name + '.jpg'

        img_path = os.path.join(img_dir, img_name + '.jpg')
        img = Image.open(img_path)
        w, h = img.size

        sample_dict['im_height'] = h
        sample_dict['im_width'] = w
        sample_dict['category_id'] = ans2label[ans] - 2
        data_dict['annotations'].append(sample_dict)

        valid_id += 1
    cnt += 1

with open(output_name, 'w') as f:
    json.dump(data_dict, f)
