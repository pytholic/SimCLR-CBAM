# Alzheimer-classification


## Preparation
```
Edit make_dataset_list. In the main function, you can edit your dataset path
$ conda env create --file environment.yaml
```
##Training parameter
```
--class_type : 'AD_CN', 'MCI_CN', AD_MCI', '3class'
--pretrained_model : type your SimCLR pretrained model path
--dataset_path : type your dataset path like '/data/tm/alzh/PGGAN_data
```
## Dataset downloading path
```
manual selection : https://drive.google.com/file/d/1B2ubVAs9GsgKYnE75laL482ighW8oo7-/view?usp=sharing
SSIM : https://drive.google.com/file/d/1O_UWQUaRlDuY9I1s5_sNbSiEWxEA7Tjo/view?usp=sharing
```

## Pretrain using SimCLR
```
$ python contrastive_main.py --batch_size 128 --dataset_path your_dataset_path
```

## Classification
```
$ python main.py --batch_size 128 --pretrained_model ./save_models/SimCLR_pretrained.pth --class_type AD_CN --dataset_path your_dataset_path
```
