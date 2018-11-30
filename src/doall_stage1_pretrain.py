from __future__ import print_function, absolute_import
import os, sys, time, datetime, argparse
import os.path as osp, numpy as np, more_itertools as mit

# import torch related packages
import torch, torch.nn as nn
from torch.optim import lr_scheduler

# import other miscellaneous things
from cmd_options_parser import stage1_parse_cmd_options
from losses import CrossEntropyLabelSmooth, TripletLoss
import utils
from utils import set_default_device, init_logger, get_currenttime_prefix, save_checkpoint
from project_utils import init_data_loaders, init_model, train, test

if __name__ == '__main__':
    # parse commandline arguments
    args = stage1_parse_cmd_options()

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
    dataset, trainloader, queryloader, galleryloader = init_data_loaders(args)
    num_train_pids = dataset.num_train_pids

    # init model
    model = init_model(args, num_train_pids)
    if use_gpu:
        model = nn.DataParallel(model).cuda()
    vis = utils.get_visdom_for_current_run(args.save_dir, args.prefix + '_stage1_training')

    # init objective functions
    criterion_xent = CrossEntropyLabelSmooth(num_classes=num_train_pids, use_gpu=use_gpu)
    criterion_htri = TripletLoss(margin=args.margin)

    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    start_epoch = args.start_epoch

    # if only evaluation was needed
    if args.evaluate:
        print("Evaluate only")
        test(model, queryloader, galleryloader, use_gpu, args)
        exit(0)

    start_time = time.time()
    best_rank1 = -np.inf

    vis = utils.get_visdom_for_current_run(args.save_dir, 'stage1_pretraining')
    for epoch in range(start_epoch, args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        
        train(vis, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu, args)
        
        if args.stepsize > 0: scheduler.step()
        
        if args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, use_gpu, args)  
            is_best = rank1 > best_rank1
            if is_best: best_rank1 = rank1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, args.prefix + '_checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
