from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import random

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]

            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, camid

        if self.sample == 'evenly':
            """
            Evenly sample seq_len items from num items.
            """
            if num >= self.seq_len:
                num -= num % self.seq_len
                indices = np.arange(0, num, int(num/self.seq_len))
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32)*(num-1)])
                
            assert len(indices) == self.seq_len

            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, camid

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index=0
            frame_indices = list(range(num))
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)

            imgs_list=[]
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                #imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))


class PairVideoDataset(VideoDataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None):
        super().__init__(dataset=dataset, seq_len=seq_len, sample=sample, transform=transform)
        
        # in dataset, each instance contains img paths, person id, cam id
        # arrange the details based on pid for pairwise training
        self.pid_based_dataset = self.arrange_based_on_pid()
        self.pids = list(self.pid_based_dataset.keys())
        print('In total ', len(self.pids), ' persons' )
    
    def arrange_based_on_pid(self):
        pid_based_dataset = {}

        #  for each instance of the dataseet, arrange it based on pid
        for index in range(len(self.dataset)):
            img_paths, pid, camid = self.dataset[index]

            # init empty buffer if key does not exist
            if pid not in pid_based_dataset:
                pid_based_dataset[pid] = {}
            
            if camid not in pid_based_dataset[pid]:
                pid_based_dataset[pid][camid] = []
            
            # store the indices in the buffer
            pid_based_dataset[pid][camid].append(index)
        
        # remove the persons that appear in only one cam
        only_one_cam_persons = []
        for pid in pid_based_dataset.keys():
            cams = list(pid_based_dataset[pid].keys())
            if len(cams) == 1:
                only_one_cam_persons.append(pid)
        
        print('these IDs are removed from training', only_one_cam_persons)
        for pid in only_one_cam_persons:
            del pid_based_dataset[pid]
        
        return pid_based_dataset

    def __len__(self):
        # for each person, consider positive and negative pairs
        return len(self.pids) * 2
    
    def __getitem__(self, index):
        # select the person
        # print(len(self.pids), index, index//2)
        pid1 = self.pids[index//2]
        camid1 = np.random.choice(list(self.pid_based_dataset[pid1].keys()))
        
        # even indices are of same pid, odd indices are different pid
        if index%2 == 0:
            # select same pid frames
            pid2 = pid1
            list_except_camid1 = list(self.pid_based_dataset[pid1].keys())
            # print(pid1, list_except_camid1)
            list_except_camid1.remove(camid1)
            # print(pid1, list_except_camid1)
            camid2 = np.random.choice(list_except_camid1)
            assert (pid1 == pid2) and (camid1 != camid2)
        else:
            # select different pid frames
            list_except_pid1 = list(self.pids)
            list_except_pid1.remove(pid1)
            pid2 = np.random.choice(list_except_pid1)
            camid2 = np.random.choice(list(self.pid_based_dataset[pid2].keys()))
            assert pid1 != pid2

        person1_img_index = np.random.choice(self.pid_based_dataset[pid1][camid1])
        person2_img_index = np.random.choice(self.pid_based_dataset[pid2][camid2])
        img_frames1, pid1_returned, camid1_returned = super().__getitem__(person1_img_index)
        img_frames2, pid2_returned, camid2_returned = super().__getitem__(person2_img_index)

        assert pid1_returned == pid1, 'pid1 equality assertion failed'
        assert camid1_returned == camid1, 'camid1 equality assertion failed'
        assert pid2_returned == pid2, 'pid2 equality assertion failed'
        assert camid2_returned == camid2, 'camid2 equality assertion failed'

        return img_frames1, pid1, img_frames2, pid2