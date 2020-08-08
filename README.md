## HCP-MIC at ImageCLEF VQA-Med 2020
This repository is the official implementation of paper [HCP-MIC at VQA-Med 2020: Effective Visual Representation for Medical Visual Quesion Answering].

## Citing this repository
If you find this code useful in your work, please consider citing us:

```
@inproceedings{chen2020hcp-mic,
	title={HCP-MIC at VQA-Med 2020: Effective Visual Representation for Medical Visual Quesion Answering},
	author={Guanqi Chen and Haifan Gong and Guanbin Li},
	booktitle={CLEF (Working notes)},
	year={2020}
}
```

## Main requirements

  * **torch == 1.4.0**
  * **torchvision == 0.5.0**
  * **tensorboardX == 2.0**
  * **Python 3**

## Pretrain models for VQA-Med 2020

We provide the pretrain models of both medical images classifier and medical questions classifier for VQA-Med 2020.
These models should be under the ```BBN-BioBert-Inference/pretrain``` folder.

[Baidu Cloud](https://pan.baidu.com/s/1LX9NZ66PLnacqhZSo7mXvg) code:93nw

The BBN is mainly modified from [BBN](https://github.com/Megvii-Nanjing/BBN), Bio-Bert pretrain is obtained from [Biobert](https://github.com/dmis-lab/biobert), the pickle data should be under the ```BBN-BioBert-Inference/data/``` folder. 

## Usage
```bash
# To train long-tailed abnormal images classification with BBN-ResNeSt-50:
python BBN/main/train.py  --cfg BBN/configs/BBN-ResNeSt-50.yaml     

# To train medical questions classification with bio-bert:
cd BioBert
python train.py

# To validate with the best model
cd BBN-BioBert-Inference
python inference.py
```

You can change the experimental setting by simply modifying the parameter in the yaml file.

## Tools

```bash
# Get json from the original format of VQA-MED 2020
python BBN/txt2json.py

# Create cache for VQA
python BBN/create_cache4VQA.py

# Create pickel for inference
python BBN/bbn_create_pickel.py

# Expanding dataset via KL divergence
python BBN/expand_dataset.py

# Create feature dict
python BBN-Biobert-Inference/create_feature_dict.py
```

## Data format

The annotation of a dataset is a dict consisting of two field: `annotations` and `num_classes`.
The field `annotations` is a list of dict with
`image_id`, `fpath`, `im_height`, `im_width` and `category_id`.

Here is an example.
```
{
    'annotations': [
                    {
                        'image_id': 1,
                        'fpath': '/home/data/train1920/images/synpic593.jpg',
                        'im_height': 600,
                        'im_width': 800,
                        'category_id': 74
                    },
                    ...
                   ]
    'num_classes': 330
}
```
You can use the following code to convert from the original format of VQA-Med. 
The images and annotations can be downloaded at [VQA-MED 2020](https://www.aicrowd.com/challenges/imageclef-2020-vqa-med-vqa)


## Contacts
If you have any questions about our work, please do not hesitate to contact us by emails.

Haifan Gong: haifangong@outlook.com
