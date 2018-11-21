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

    vis = utils.get_visdom_for_current_run('funcapprox_td0_qlearning')

    # average meters for losses and rewards
    avg_datacoll_reward_meter = utils.AverageMeter(vis, 'data collection rewards', 'epoch', 'rewards')
    avg_train_loss_meter = utils.AverageMeter(vis, 'train loss', 'epoch', 'loss')
    avg_test_reward_meter = utils.AverageMeter(vis, 'test rewards', 'epoch', 'rewards')
    
    for epoch in tqdm(range(args.max_epoch)):
        # in each epoch, collect data
        average_data_collection_reward = model.collect_data(trainloader, args)
        # train the network
        average_train_loss = model.train(args=args, num_runs=args.num_train_iterations)
        # test the network
        average_test_reward = model.test(args.num_test_iterations)

        # logistics
        print(average_data_collection_reward, average_train_loss, average_test_reward)
        avg_datacoll_reward_meter.update(average_data_collection_reward, len(trainloader))
        avg_train_loss_meter.update(average_train_loss, args.num_train_iterations)
        avg_test_reward_meter.update(average_test_reward, args.num_test_iterations)    

    # save video, if needed
    if 'monitor' in sys.argv:
        print('plotting and saving the video')
        filename = os.path.basename(__file__).split(".")[0]
        monitor_dir = "../scratch/" + filename + "_" + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)
    
    average_test_reward = model.test(1)
    print('reward in testing: ', average_test_reward)
    env.env.close()
    env.close()
