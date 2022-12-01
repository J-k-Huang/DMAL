from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from utils import *
from taskcv_loader import CVDataLoader
from basenet1 import *
import torch.nn.functional as F
import os
from torch.nn.parallel import DataParallel
import scipy.io as sio 
from solver import Solver
#from aligned_reid.utils.utils import set_devices

# Training settings
parser = argparse.ArgumentParser(description='DomainNet Classification')
parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='source only or not')
parser.add_argument('--batch-size', type=int, default=24, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=24, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                    help='the name of optimizer')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--eval_only', action='store_true', default=False,
                    help='evaluation only option')
parser.add_argument('--one_step', action='store_true', default=False,
                    help='one step training with gradient reversal layer')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_k', type=int, default=1, metavar='K',
                    help='how many steps to repeat the generator update')
parser.add_argument('--num-layer', type=int, default=2, metavar='K',
                    help='how many layers for classifier')
parser.add_argument('--name', type=str, default='board', metavar='B',
                    help='board dir')
parser.add_argument('--save', type=str, default='/media/zrway/8T/HJK/DMAL/Image/save_model/', metavar='B',
                    help='board dir')

parser.add_argument('--train_path', type=str, default='/media/zrway/8T/HJK/aa/DomainNet/M3SDA/painting', metavar='B',
                    help='directory of source datasets')
parser.add_argument('--val_path', type=str, default='/media/zrway/8T/HJK/aa/DomainNet/M3SDA/sketch', metavar='B',
                    help='directory of target datasets')
parser.add_argument('--resnet', type=str, default='101', metavar='B',help='which resnet 18,50,101,152,200')
parser.add_argument('--task_name', type=str, default='t-v', metavar='B',help='domain1-domain2')
parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
#-------
#i-p-c
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1,2'
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']='3'
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)


def main():
    # if not args.one_step:

    solver = Solver(args, train_path = args.train_path, val_path = args.val_path, learning_rate=args.lr, batch_size=args.batch_size,
                    optimizer=args.optimizer, num_k=args.num_k,
                    checkpoint_dir=args.checkpoint_dir)
    record_num = 0    
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists('record'):
        os.mkdir('record')
    if args.eval_only:
        solver.test(0)
    else:
        count = 0 
        
        acc=0
        for t in range(args.epochs):
            num = solver.train(t)
            #count += num
            if t % 1 == 0:
                acc = solver.test(t,acc)
            if count >= 20000:
                break
            #if args.source == 'svhn' and t>=7:
            	#break
if __name__ == '__main__':
    main()


