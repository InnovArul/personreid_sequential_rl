from __future__ import print_function, absolute_import
import os, sys, time, datetime, argparse
import os.path as osp, numpy as np, more_itertools as mit

# import torch related packages
import torch, torch.nn as nn
from torch.optim import lr_scheduler

# import other miscellaneous things
from cmd_options_parser import stage2_parse_cmd_options
from losses import CrossEntropyLabelSmooth, TripletLoss
import utils
from utils import set_default_device, init_logger, get_currenttime_prefix, save_checkpoint
from project_utils import init_data_loaders_rl_training, init_model_rl_training, train, test
from tqdm import tqdm

if __name__ == '__main__':
    # parse commandline arguments
    args = stage2_parse_cmd_options()

    # set the seed for reproducibility
    torch.manual_seed(args.seed)
    use_gpu = set_default_device(args)

    # append date with save_dir
    args.save_dir = '../scratch/' + get_currenttime_prefix() + '_' + args.save_dir
    if args.pretrained_model is not None:
        args.save_dir = os.path.dirname(args.pretrained_model)

    # init the logger
    init_logger(args)

    # data loading
    dataset, trainloader, queryloader, galleryloader = init_data_loaders_rl_training(args)
    num_train_pids = dataset.num_train_pids

    # init model
    model = init_model_rl_training(args, num_train_pids)
    vis = utils.get_visdom_for_current_run(args.save_dir, 'stage2_rl_training')

    # average meters for losses and rewards
    avg_datacoll_reward_meter = utils.AverageMeter(vis, 'data collection rewards', 'epoch', 'rewards')
    avg_train_loss_meter = utils.AverageMeter(vis, 'train loss', 'epoch', 'loss')
    avg_test_reward_meter = utils.AverageMeter(vis, 'test rewards', 'epoch', 'rewards')
    
    for epoch in tqdm(range(args.start_epoch, args.max_epoch)):
        print('epoch #', epoch)
        # in each epoch, collect data
        print("collecting data\n")
        average_data_collection_reward = model.collect_data(trainloader, epoch, args)
        # train the network
        print("training\n")
        average_train_loss = model.train(args=args, num_runs=args.num_train_iterations)

        # logistics
        print(average_data_collection_reward, average_train_loss)
        avg_datacoll_reward_meter.update(average_data_collection_reward, len(trainloader))
        avg_train_loss_meter.update(average_train_loss, args.num_train_iterations)   

        if (epoch+1) % args.save_step == 0 or (epoch+1) == args.max_epoch:
            filename = osp.join(args.save_dir, args.prefix + '_checkpoint_ep' + str(epoch+1) + '.pth.tar')
            state_dict = model.state_dict()
            print('saving checkpoint to : ', filename) 

            save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch,
            }, True, filename)

            # test the network
            print("testing\n")
            cmc = model.test(queryloader, galleryloader, args)
            avg_test_reward_meter.update(cmc, 1) 
            print(average_data_collection_reward, average_train_loss, cmc)
