import os, sys, argparse
import data_manager

def stage1_parse_cmd_options():
    parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
    # Datasets
    parser.add_argument('-d', '--dataset', type=str, default='mars',
                        choices=data_manager.get_names())
    parser.add_argument('-j', '--workers', default=0, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--height', type=int, default=224,
                        help="height of an image (default: 224)")
    parser.add_argument('--width', type=int, default=112,
                        help="width of an image (default: 112)")
    parser.add_argument('--seq-len', type=int, default=4, help="number of images to sample in a tracklet")
    parser.add_argument('--test-num-tracks', type=int, default=16, help="number of tracklets to pass to GPU during test (to avoid OOM error)")
    
    # Optimization options
    parser.add_argument('--max-epoch', default=800, type=int,
                        help="maximum epochs to run")
    parser.add_argument('--start-epoch', default=0, type=int,
                        help="manual epoch number (useful on restarts)")
    parser.add_argument('--train-batch', default=32, type=int,
                        help="train batch size")
    parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
    parser.add_argument('--stepsize', default=200, type=int,
                        help="stepsize to decay learning rate (>0 means this is enabled)")
    parser.add_argument('--gamma', default=0.1, type=float,
                        help="learning rate decay")
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")
    parser.add_argument('--margin', type=float, default=0.9, help="margin for triplet loss")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="number of instances per identity")
    
    # Architecture
    parser.add_argument('-a', '--arch', type=str, default='alexnet', help="resnet50, alexnet")

    # Miscs
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--print-freq', type=int, default=77, help="print frequency")
    parser.add_argument('--seed', type=int, default=1, help="manual seed")
    parser.add_argument('--pretrained-model', type=str, default=None, help='need to be set for resnet3d models')
    parser.add_argument('--evaluate', action='store_true', help="evaluation only")
    parser.add_argument('--eval-step', type=int, default=50,
                        help="run evaluation for every N epochs (set to -1 to test after training)")
    parser.add_argument('--save-dir', type=str, default='multishot-rl')
    parser.add_argument('--use-cpu', action='store_true', help="use cpu")
    parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    return args

def stage2_parse_cmd_options():
    parser = argparse.ArgumentParser(description='Train RL model')

    # Datasets
    parser.add_argument('-d', '--dataset', type=str, default='mars',
                        choices=data_manager.get_names())
    parser.add_argument('-j', '--workers', default=0, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--height', type=int, default=224,
                        help="height of an image (default: 224)")
    parser.add_argument('--width', type=int, default=112,
                        help="width of an image (default: 112)")
    parser.add_argument('--seq-len', type=int, default=4, help="number of images to sample in a tracklet")
    parser.add_argument('--rl-seq-len', type=int, default=8, help="number of images to sample in a tracklet")
    parser.add_argument('--test-num-tracks', type=int, default=16, help="number of tracklets to pass to GPU during test (to avoid OOM error)")
    
    # Optimization options
    parser.add_argument('--max-epoch', default=800, type=int, help="maximum epochs to run")
    parser.add_argument('--start-epoch', default=0, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument('--train-batch', default=1, type=int, help="train batch size")
    parser.add_argument('--test-batch', default=1, type=int, help="has to be 1") 
    parser.add_argument('--num-train-iterations', default=100000, type=int, help="train iterations") 
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
    parser.add_argument('--stepsize', default=400, type=int, help="stepsize to decay learning rate (>0 means this is enabled)")
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
    parser.add_argument('--weight-decay', default=5e-04, type=float, help="weight decay (default: 5e-04)")
    parser.add_argument('--num-instances', type=int, default=4, help="number of instances per identity")
    
    # Architecture
    parser.add_argument('-a', '--arch', type=str, default='alexnet', help="resnet50, alexnet")
    parser.add_argument('--rp', type=float, default=0.2, help="reward per step")

    # Miscs
    parser.add_argument('--rl-algo', type=str, default='ql', help='ql - Q learning, pg - Policy gradient')
    parser.add_argument('--prefix', type=str, default='rl')
    parser.add_argument('--print-freq', type=int, default=77, help="print frequency")
    parser.add_argument('--seed', type=int, default=1, help="manual seed")
    parser.add_argument('--pretrained-model', type=str, default=None, help='need to be set for loading pretrained models')
    parser.add_argument('--pretrained-model-rl', type=str, default=None, help='need to be set for loading pretrained rl models')
    parser.add_argument('--evaluate', action='store_true', help="evaluation only")
    parser.add_argument('--save-step', type=int, default=50, help="save model for every N epochs (set to -1 to test after training)")
    parser.add_argument('--save-dir', type=str, default='multishot-rl')
    parser.add_argument('--use-cpu', action='store_true', help="use cpu")
    parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    return args