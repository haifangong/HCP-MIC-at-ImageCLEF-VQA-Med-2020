import json
import os
import pickle

import numpy as np
import cv2
from torchvision import transforms
from tqdm import tqdm

def get_image(fpath):
    img = cv2.imread(fpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def shorter_resize_for_crop(size):
    return transforms.Resize(int(size[0] / 0.875))


def center_crop(size):
    return transforms.CenterCrop(size)


def normalize():
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )


def update_transform(input_size=(224, 224)):
    transform_list = [transforms.ToPILImage(), shorter_resize_for_crop(input_size), center_crop(input_size)]
    transform_list.extend([transforms.ToTensor(), normalize()])
    transform = transforms.Compose(transform_list)
    return transform


def bbn_images2pkl(dtype='test', ytype='2020'):
    size = 224
    img_dir = dtype + ytype + '/clipimages/'
    img_id2idx_path = '/home/duadua/code-for-haifan/BBN-BioBert-Inference/data/trainimgid2idx.json'
    img_id2idx = json.load(open(img_id2idx_path))
    img_names = list(img_id2idx.keys())
    final_np = np.zeros((len(img_id2idx), 3, size, size), dtype=np.float32)
    transform = update_transform()

    for img_name in tqdm(img_names):
        img_path = os.path.join(img_dir, img_name)
        img = get_image(img_path)
        img = transform(img)
        img_np = img.numpy()
        # print(img_np.shape)

        final_np[img_id2idx[img_name]] = img_np

    with open(dtype + str(size) + 'x' + str(size) + 'clip.pkl', 'wb') as f:
        pickle.dump(final_np, f)


if __name__ == "__main__":
    bbn_images2pkl(dtype='train', ytype='1920')
