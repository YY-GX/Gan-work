import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter   
import numpy as np

# Import my custom dataset
from rgrDataset import RgrDataset
from smallNet import smallNet
import csv

from MultiTaskModel import MultiTaskModel, MultiTaskModel1, MultiTaskModel2, MultiTaskModel3, MultiTaskModel4
from MultiTaskDataset import MultiTaskDataset, MultiTaskPETDataset

from datetime import datetime,timezone,timedelta


SEED=1

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic=True


# TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# My own params >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
parser.add_argument('--logpath', default='logs/log_raw', type=str,
                    help='Path to store tensorboard log files.')
parser.add_argument('--traindatapath', default=None, type=str,
                    help='Path of the train data.')
parser.add_argument('--testdatapath', default=None, type=str,
                    help='Path of the test data.')
parser.add_argument('--augement', default=None, type=str,
                    help='choose pure or toge or expe')
parser.add_argument('--augedir', default=None, type=str,
                    help='if augement is toge, add the dir at the same time')
parser.add_argument('--usesmall', default=0, type=int,
                    help='1 for use and 0 for not use')
parser.add_argument('--findlr', default=None, type=int,
                    help='any type for find mode')
parser.add_argument('--resumedir', default='', type=str, metavar='PATH',
                    help='path of DIR to latest checkpoint (default: none)')
parser.add_argument('--dropout', default=0, type=float,
                    help='dropout ratio')
parser.add_argument('--times', default=1, type=int,
                    help='run the whole file for times')
parser.add_argument('--filename', default='save_number_file.csv', type=str,
                    help='file to save loss')
parser.add_argument('--lambda1', default=0.5, type=float,
                    help='coefficient of mr')
parser.add_argument('--lambda2', default=0.5, type=float,
                    help='coefficient of ct')
parser.add_argument('--mymodel', default=None, type=str,
                    help='model')


parser.add_argument('--test-dir', type=str, default='./../datasets/final_mr_ct/testB/',
                    help='path to the test folder, each class has a single folder')
best_loss = 0


args = parser.parse_args()
writer = SummaryWriter(args.logpath)



dt = datetime.utcnow()
dt = dt.replace(tzinfo=timezone.utc) 
tzutc_8 = timezone(timedelta(hours=8))
TIMESTAMP = str(dt.astimezone(tzutc_8))
dirName=args.resumedir + 'checkpoint-' + TIMESTAMP + '/'
os.mkdir(dirName)


def main():
#     args = parser.parse_args()
    
#     Read my args >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    if args.traindatapath is None or args.testdatapath is None:
        print('[Error!] Enter the data(train & test) path')

        
        
        
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_loss
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
#     if args.usesmall == 1:
# #         model = None 
# #         if args.dropout:
#         model = smallNet(args.dropout)
# #         else:
# #             model = smallNet()  #Assign smallnet 
# #         model.cuda()
#     elif args.pretrained:
#         print("=> using pre-trained model '{}'".format(args.arch))
# #         model = models.__dict__[args.arch](pretrained=True)
#         model = models.resnet18()
# #         model.fc = nn.Linear(2048, 1)
#     else:
#         print("=> creating model '{}'".format(args.arch))
# #         model = models.__dict__[args.arch]()
#         model = models.resnet18()
# #         model.fc = nn.Linear(2048, 1)
    if args.mymodel == '1':
        model = MultiTaskModel1()
    elif args.mymodel == '2':
        model = MultiTaskModel2()
    elif args.mymodel == '3':
        model = MultiTaskModel3() 
    elif args.mymodel == '4':
#         checkpoint = torch.load('./checkpoints/checkpoints_baseline_multitask/model_best.pth.tar')
        model = MultiTaskModel4()
#         model.load_state_dict(checkpoint, strict=False)
    else:
        model = MultiTaskModel()

#     if args.distributed:
#         # For multiprocessing distributed, DistributedDataParallel constructor
#         # should always set the single device scope, otherwise,
#         # DistributedDataParallel will use all available devices.
#         if args.gpu is not None:
#             torch.cuda.set_device(args.gpu)
#             model.cuda(args.gpu)
#             # When using a single GPU per process and per
#             # DistributedDataParallel, we need to divide the batch size
#             # ourselves based on the total number of GPUs we have
#             args.batch_size = int(args.batch_size / ngpus_per_node)
#             args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
#             model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
#         else:
#             model.cuda()
#             # DistributedDataParallel will divide and allocate batch_size to all
#             # available GPUs if device_ids are not set
#             model = torch.nn.parallel.DistributedDataParallel(model)
#     elif args.gpu is not None:
#         torch.cuda.set_device(args.gpu)
#         model = model.cuda(args.gpu)
#     else:
#         # DataParallel will divide and allocate batch_size to all available GPUs
#         if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
#             model.features = torch.nn.DataParallel(model.features)
#             model.cuda()
#         else:
#             model = torch.nn.DataParallel(model).cuda()

    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device {}".format(device))
            
    # Data parallel for multiple GPU usage
    if torch.cuda.device_count() > 1:
        print("[INFO] Let's use ", torch.cuda.device_count(), " GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda(args.gpu)

#     optimizer = torch.optim.SGD(model.parameters(), args.lr,
#                                 momentum=args.momentum,
#                                 weight_decay=args.weight_decay)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_loss = best_loss.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
#     traindir = 'datasets/pure/train/'
#     valdir = 'datasets/pure/val/'
    traindir = args.traindatapath
    valdir = args.testdatapath
    
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    
    root_dir_results = None
    if args.augement == 'toge':
        if args.augedir == None:
            print('Please add the augement DIR!')
            return
        else:
            root_dir_results = args.augedir

        # Train data augmentations
    train_trans = transforms.Compose(
                        [
                            transforms.Resize(256),
#                             transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            # auto.ImageNetPolicy(),
                            transforms.ToTensor(),
                            normalize
                        ]
                    )
    
    train_dataset = MultiTaskPETDataset('./pet.csv', './ct.csv', 
                                     '../datasets/final_mr_ct/pet_fake', '../datasets/final_mr_ct/trainB',
                                    transform = train_trans)
        
#     train_dataset = RgrDataset(
#         'ct.csv', traindir, args.augement, 
#         transforms.Compose([
#               transforms.Resize(256),
# #               transforms.CenterCrop(224), 
# #               transforms.RandomRotation(15),
# #               transforms.RandomAffine(degrees=15),
#               transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ]), root_dir_results = root_dir_results)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_test_trans = transforms.Compose(
                        [
                            transforms.Resize(256),
#                             transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize
                        ]
                    )

    
    val_loader = torch.utils.data.DataLoader(
        RgrDataset(
        'ct.csv', valdir, 'pure', 
        val_test_trans), 
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    

        
    testset = RgrDataset('ct.csv', args.test_dir, args.augement, transform=val_test_trans)
 
    test_loader = torch.utils.data.DataLoader(testset,batch_size=args.batch_size)

    
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    with open(args.filename, "w") as csvfile: 
        writer1 = csv.writer(csvfile)
        for i in range(args.times):
            outcome = 0
            best_loss = 9999999
            for epoch in range(args.start_epoch, args.epochs):
                if args.distributed:
                    train_sampler.set_epoch(epoch)
                lr = adjust_learning_rate(optimizer, epoch, args)

                # train for one epoch
                loss_tr_mr, loss_tr_ct, loss_tr_total = train(train_loader, model, criterion, optimizer, epoch, args)

                # evaluate on validation set
                loss = validate(val_loader, model, criterion, args, epoch)

                writer.add_scalars('Loss' + str(i), {
                    'Train_mr': loss_tr_mr,
                    'Train_ct': loss_tr_ct,
                    'Train_total': loss_tr_total,
                    'Val': loss
                }, epoch)
                writer.flush()

                # remember best acc@1 and save checkpoint
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)
                
                if (args.epochs - epoch) <= 10:
                    outcome += loss

                if is_best and epoch % 10 != 0:
                    save_checkpoint(model.state_dict(), is_best, epoch, optimizer)

                if epoch % 10 == 0:
                    save_checkpoint(model.state_dict(), is_best, epoch, optimizer)
              
            # Start test
            print('...........Testing Start..........')
            print('Loading best checkpoint ...')

            # Load best checkpont
            load_resume(args, model, optimizer, args.resumedir)

            # Test the best checkpoint
            loss_test = validate(test_loader, model, criterion, args, epoch)

            # Record the test loss
            record = {
                'loss_test': loss_test,
            }
            torch.save(record, os.path.join(args.resumedir, 'test_info.pth.tar'))

            print('[INFO] TEST LOSS: ' + str(loss_test))
            print('...........Testing End..........')
            
                    
            outcome /= 10    
            writer1.writerow(str(outcome))
#             model = models.__dict__[args.arch]()
            model = models.resnet18()
            model.fc = nn.Linear(2048, 1)
            model = model.cuda(args.gpu)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
            criterion = nn.MSELoss().cuda(args.gpu)
    #             if not args.multiprocessing_distributed or (args.multiprocessing_distributed
    #                     and args.rank % ngpus_per_node == 0):
    #                 if epoch % 10 == 0:
    #                     save_checkpoint({
    #                         'epoch': epoch + 1,
    #                         'arch': args.arch,
    #                         'state_dict': model.state_dict(),
    #                         'best_loss': best_loss,
    #                         'optimizer' : optimizer.state_dict(),
    #                     }, is_best, epoch)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_mr = AverageMeter('Loss', ':.4e')
    losses_ct = AverageMeter('Loss', ':.4e')
    losses_total = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_ct],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    
    

    end = time.time()
    for i, (image_ct, label_ct, image_mr, label_mr) in enumerate(train_loader):
#         print("[DEBUG] image_ct size: " + str(image_ct.size()) + ", label size: " + str(label_ct.size()))
#         print("[DEBUG] image_mr size: " + str(image_mr.size()) + ", label size: " + str(label_mr.size()))
#         print('images.size: ' + str(images.size()))
#         print('target.size: ' + str(target.size()))
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            image_ct = image_ct.cuda(args.gpu, non_blocking=True)
            image_mr = image_mr.cuda(args.gpu, non_blocking=True)
        else:
            image_ct = image_ct.cuda(0, non_blocking=True)
            image_mr = image_mr.cuda(0, non_blocking=True)
        label_ct = label_ct.type(torch.FloatTensor).view(-1, 1).cuda(args.gpu, non_blocking=True)
        label_mr = label_mr.type(torch.FloatTensor).view(-1, 1).cuda(args.gpu, non_blocking=True)
    

        # compute output
        output_mr, output_ct = model(image_mr, image_ct)
#         print('OUTPUT:')
#         print(output)
#         print('========================================')
#         print('TARGET:')
#         print(target)
        loss_mr = criterion(output_mr, label_mr)
        loss_ct = criterion(output_ct, label_ct)
        loss_total = args.lambda1 * loss_mr + args.lambda2 * loss_ct

        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses_mr.update(loss_mr.item(), image_mr.size(0))
        losses_ct.update(loss_ct.item(), image_ct.size(0))
        losses_total.update(loss_total.item(), image_ct.size(0))
        # top1.update(acc1[0], images.size(0))
        # top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            
    
    return losses_mr.avg, losses_ct.avg, losses_total.avg


def validate(val_loader, model, criterion, args, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.view(-1, 1).type(torch.FloatTensor).cuda(args.gpu, non_blocking=True)
            # compute output
            output_ct = model(None, images)
            loss = criterion(output_ct, target)
            
            # measure accuracy and record loss
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            # top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' MSE:{loss}'
              .format(loss=loss))
        
#     writer.add_scalar('Test/Loss', losses.avg, epoch)
#     writer.flush()
    return losses.avg

# Util function: load resume
def load_resume(args, model, optimizer, load_path):
    if load_path:
        # Default load best.pth.tar
        load_path = os.path.join(load_path, 'model_best.pth.tar')
        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))
            checkpoint = torch.load(load_path)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(load_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(load_path))
    else:
        print('[ERROR] No load path provided ...') 
    

# def save_checkpoint(state, is_best, epoch, filename=args.resumedir):
#     filename = args.resumedir + 'checkpoint-' + TIMESTAMP + '/' + str(epoch) + '.pth.tar'
#     print(filename)
# #     filename = args.resumedir
#     print('==================================================')
#     print('= ', filename)
#     print('==================================================')
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')

def save_checkpoint(state, is_best, epoch, optimizer, checkpoint_path=args.resumedir):
    record = {
        'epoch': epoch + 1,
        'state_dict': state,
        'optimizer': optimizer.state_dict(),
    }
    filename = os.path.join(checkpoint_path, 'record_epoch{}.pth.tar'.format(epoch))
    torch.save(record, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_path, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.findlr:
        lr = args.lr * 1.05
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res


if __name__ == '__main__':
    main()
