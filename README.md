# personreid sequential rl

An independent attempt to replicate the paper "Multi-shot Pedestrian Re-identification via Sequential Decision Making (CVPR2018)"

## Prerequisites

```
torchvision
torch >= 0.4.0
visdom
tqdm
more-itertools
```

## Data preparation
To prepare data under './data' folder, refer [DATASET](https://github.com/KaiyangZhou/deep-person-reid/blob/master/DATASETS.md) preparation from [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) repository. 

### MARS [8]:

1. Create a directory named mars/ under data/.
2. Download dataset to data/mars/ from http://www.liangzheng.com.cn/Project/project_mars.html.
3. Extract bbox_train.zip and bbox_test.zip.
4. Download split information from https://github.com/liangzheng06/MARS-evaluation/tree/master/info and put info/ in data/mars (we want to follow the standard split in [8]). The data structure would look like:

```
mars/
    bbox_test/
    bbox_train/
    info/
```

Use `-d mars` when running the training code.

### iLIDS-VID [11]:

The code supports automatic download and formatting. Simple use `-d ilidsvid` when running the training code. The data structure would look like:

```
ilids-vid/
    i-LIDS-VID/
    train-test people splits/
    splits.json
```

### PRID [12]:

1. Under data/, do mkdir prid2011 to create a directory.
2. Download dataset from https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/ and extract it under data/prid2011.
3. Download the split created by iLIDS-VID from here, and put it in data/prid2011/. We follow [11] and use 178 persons whose sequences are more than a threshold so that results on this dataset can be fairly compared with other approaches. The data structure would look like:

```
prid2011/
    splits_prid2011.json
    prid_2011/
        multi_shot/
        single_shot/
        readme.txt
```
Use `-d prid` when running the training code.

### DukeMTMC-VideoReID [16, 23]:

1. Make a directory data/dukemtmc-vidreid.
2. Download dukemtmc_videoReID.zip from https://github.com/Yu-Wu/DukeMTMC-VideoReID. Unzip the file to data/dukemtmc-vidreid. 

You need to have

```
dukemtmc-vidreid/
    dukemtmc_videoReID/
        train_split/
        query_split/
        gallery_split/
        ... (and two license files)
```
Use `-d dukemtmcvidreid` when running the training code.


## Options


## Credits
Github repositories: 

* [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)
* [Video-Person-ReID](https://github.com/jiyanggao/Video-Person-ReID)

