# general utils
import os, sys
from tqdm import tqdm
from utils import get_learnable_params, load_pretrained_model
import torch
from torch.utils.data import DataLoader
from eval_metrics import evaluate
from utils import AverageMeter, Logger, save_checkpoint, get_features
import numpy as np

# import data managers
import data_manager
from video_loader import VideoDataset, PairVideoDataset
import transforms as T
from samplers import RandomIdentitySampler

# import models
import models
from models.alexnet import AlexNet
from models.ResNet50 import ResNet50TP
from models.RL_model import Agent as Agent_QL
from models.RL_model_policygradient import Agent as Agent_PG

def init_data_loaders(args, use_gpu=True):
    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset)

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False


    trainloader = DataLoader(
        VideoDataset(dataset.train, seq_len=args.seq_len, sample='random',transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        VideoDataset(dataset.query, seq_len=args.seq_len, sample='random', transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, seq_len=args.seq_len, sample='random', transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    return dataset, trainloader, queryloader, galleryloader

def init_data_loaders_rl_training(args, use_gpu=True, test_shuffle=False):
    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset)

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        PairVideoDataset(dataset.train, seq_len=args.seq_len, sample='random',transform=transform_train),
        # sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        VideoDataset(dataset.query, seq_len=1, sample='dense', transform=transform_test),
        batch_size=args.test_batch, shuffle=test_shuffle, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, seq_len=1, sample='dense', transform=transform_test),
        batch_size=args.test_batch, shuffle=test_shuffle, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    return dataset, trainloader, queryloader, galleryloader

def init_model(args, num_train_pids):

    print("Initializing model: {}".format(args.arch))
    if args.arch.lower() =='resnet50':
        model = ResNet50TP(num_classes=num_train_pids)
    elif args.arch.lower() =='alexnet':
        model = AlexNet(num_classes=num_train_pids)
    else:
        assert False, 'unknown model ' + args.arch

    # pretrained model loading
    if args.pretrained_model is not None:
        model = load_pretrained_model(model, args.pretrained_model)
    
    return model


def init_model_rl_training(args, num_train_pids):
    base_model = init_model(args, num_train_pids)

    if args.rl_algo == 'ql':
        print('creating agent for Q learning')
        agent_model = Agent_QL(base_model, args)
    elif args.rl_algo == 'pg':
        print('creating agent for Policy gradient')
        agent_model = Agent_PG(base_model, args)
    else:
        assert False, 'unknown rl algo ' + args.rl_algo

    # pretrained model loading
    if args.pretrained_model_rl is not None:
        agent_model = load_pretrained_model(agent_model, args.pretrained_model_rl)
    
    return agent_model

def train(vis, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu, args):
    model.train()
    losses = AverageMeter(vis, 'loss vs. iterations', 'loss', 'iterations')
    total_loss = 0

    for batch_idx, (imgs, pids, _) in enumerate(tqdm(trainloader)):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
        
        outputs, video_features, _ = model(imgs)

        # combine hard triplet loss with cross entropy loss
        xent_loss = criterion_xent(outputs, pids)
        htri_loss = criterion_htri(video_features, pids)
        loss = xent_loss + htri_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), pids.size(0))
        total_loss += loss.item()

        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))
    
    print('total loss:', total_loss)


def test(model, queryloader, galleryloader, use_gpu, args, ranks=[1, 5, 10, 20]):
    model.eval()

    qf, q_pids, q_camids = [], [], []
    print('extracting query feats')
    for batch_idx, (imgs, pids, camids) in enumerate(tqdm(queryloader)):
        if use_gpu:
            imgs = imgs.cuda()

        with torch.no_grad():
            #imgs = Variable(imgs, volatile=True)
            # b=1, n=number of clips, s=16
            if imgs.ndimension() <= 5:
                imgs = imgs.unsqueeze(0)

            b, n, s, c, h, w = imgs.size()
            assert(b==1)
            imgs = imgs.view(b*n, s, c, h, w)
            features = get_features(model, imgs, args.test_num_tracks)
            features = torch.mean(features, 0)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
            torch.cuda.empty_cache()

    qf = torch.stack(qf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

    gf, g_pids, g_camids = [], [], []
    print('extracting gallery feats')
    for batch_idx, (imgs, pids, camids) in enumerate(tqdm(galleryloader)):
        if use_gpu:
            imgs = imgs.cuda()

        with torch.no_grad():
            if imgs.ndimension() <= 5:
                imgs = imgs.unsqueeze(0)
                
            b, n, s, c, h, w = imgs.size()
            imgs = imgs.view(b*n, s , c, h, w)
            assert(b==1)

            # handle chunked data
            features = get_features(model, imgs, args.test_num_tracks)
            # take average of features
            features = torch.mean(features, 0)
            
            torch.cuda.empty_cache()

        features = features.data.cpu()
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)

    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    return cmc[0]
