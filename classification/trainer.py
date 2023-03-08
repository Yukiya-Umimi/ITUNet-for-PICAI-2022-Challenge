import math
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tr
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import transforms

from classification.data_utils.data_loader import DataGenerator
from classification.data_utils.transforms import RandomRotate
from classification.utils import dfs_remove_weight


class Classifier(object):
    '''
    Control the training, evaluation, test and inference process.
    Args:
    - net_name: string, __all__ = [''].
    - lr: float, learning rate.
    - n_epoch: integer, the epoch number
    - channels: integer, the channel number of the input
    - num_classes: integer, the number of class
    - input_shape: tuple of integer, input dim
    - batch_size: integer
    - num_workers: integer, how many subprocesses to use for data loading.
    - device: string, use the specified device
    - pre_trained: True or False, default False
    - weight_path: weight path of pre-trained model
    '''
    def __init__(self, net_name=None, gamma=0.1, lr=1e-3, n_epoch=1, channels=1, num_classes=3, input_shape=None,
                 batch_size=6, num_workers=0, device=None, pre_trained=False, weight_path=None, weight_decay=0.,
                 mean=(0.105393,), std=(0.203002,), milestones=None,use_fp16=False,
                 external_pretrained=False):
        super(Classifier, self).__init__()

        self.net_name = net_name
        self.lr = lr
        self.n_epoch = n_epoch
        self.channels = channels
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        self.pre_trained = pre_trained
        self.weight_path = weight_path
        self.start_epoch = 0
        self.global_step = 0
        self.loss_threshold = 1.0
        self.metric = 0.0
        # save the middle output
        self.feature_in = []
        self.feature_out = []
        self.weight_decay = weight_decay
        self.mean = mean
        self.std = std
        self.gamma = gamma
        self.milestones = milestones
        self.use_fp16 = use_fp16
        self.external_pretrained = external_pretrained

        os.environ['CUDA_VISIBLE_DEVICES'] = self.device

        self.net = self._get_net(self.net_name)
        if self.pre_trained:
            self._get_pre_trained(self.weight_path)
            self.loss_threshold = eval(os.path.splitext(
                self.weight_path.split(':')[-1])[0])

        self.transform = [
            tr.Resize(size=self.input_shape),  #2
            tr.RandomAffine(0,(0.05,0.05),(0.8,1.2)),  #3
            tr.ColorJitter(brightness=.3, hue=.3, contrast=.3),  #4
            tr.RandomPerspective(distortion_scale=0.6, p=0.5),  #5
            RandomRotate((-12,12)),
            tr.RandomHorizontalFlip(p=0.5),   #  7
            tr.RandomVerticalFlip(p=0.5),   #8
            tr.ToTensor(),   #9
        ]

    def trainer(self, train_path, val_path, label_dict, cur_fold, output_dir=None, log_dir=None, optimizer='Adam',
                loss_fun='Cross_Entropy', class_weight=None, lr_scheduler=None):

        torch.manual_seed(0)
        print('Device:{}'.format(self.device))
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        output_dir = os.path.join(output_dir, "fold"+str(cur_fold))
        log_dir = os.path.join(log_dir, "fold"+str(cur_fold))

        if os.path.exists(log_dir):
            if not self.pre_trained:
                shutil.rmtree(log_dir)
                os.makedirs(log_dir)
        else:
            os.makedirs(log_dir)

        if os.path.exists(output_dir):
            if not self.pre_trained:
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
        else:
            os.makedirs(output_dir)

        self.writer = SummaryWriter(log_dir)
        self.global_step = self.start_epoch * \
            math.ceil(len(train_path)/self.batch_size)

        net = self.net
        lr = self.lr
        loss = self._get_loss(loss_fun, class_weight)
        weight_decay = self.weight_decay

        if len(self.device.split(',')) > 1:
            net = DataParallel(net)

        train_transformer = transforms.Compose(self.transform)

        train_dataset = DataGenerator(
            train_path, label_dict, channels=self.channels, transform=train_transformer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        # copy to gpu
        net = net.cuda()
        loss = loss.cuda()

        # optimizer setting
        optimizer = self._get_optimizer(optimizer, net, lr, weight_decay)
        scaler = GradScaler()

        if lr_scheduler is not None:
            lr_scheduler = self._get_lr_scheduler(lr_scheduler, optimizer)

        # acc_threshold = 0.5
        min_loss = 1.
        max_acc = 0.


        early_stopping = EarlyStopping(patience=50,verbose=True,monitor='val_ap',op_type='max')
        for epoch in range(self.start_epoch, self.n_epoch):
            train_loss, train_acc = self._train_on_epoch(epoch, net, loss, optimizer, train_loader,scaler)

            torch.cuda.empty_cache()

            val_loss, val_ap = self._val_on_epoch(epoch, net, loss, val_path, label_dict)

            if lr_scheduler is not None:
                lr_scheduler.step()

            print('Train epoch:{},train_loss:{:.5f},train_acc:{:.5f}'
                  .format(epoch, train_loss, train_acc))

            print('Val epoch:{},val_loss:{:.5f},val_ap:{:.5f}'
                  .format(epoch, val_loss, val_ap))

            self.writer.add_scalars(
                'data/loss', {'train': train_loss, 'val': val_loss}, epoch
            )
            self.writer.add_scalars(
                'data/acc', {'train': train_acc, 'val': val_ap}, epoch
            )
            self.writer.add_scalar(
                'data/lr', optimizer.param_groups[0]['lr'], epoch
            )

            early_stopping(val_ap)
            
            if val_ap > self.metric:
                self.loss_threshold = val_loss
                self.metric = val_ap
                
                min_loss = min(min_loss, val_loss)
                max_acc = max(max_acc, val_ap)

                if len(self.device.split(',')) > 1:
                    state_dict = net.module.state_dict()
                else:
                    state_dict = net.state_dict()

                saver = {
                    'epoch': epoch,
                    'save_dir': output_dir,
                    'state_dict': state_dict,
                }

                file_name = 'epoch:{}-train_loss:{:.5f}-val_loss:{:.5f}-train_acc:{:.5f}-val_ap:{:.5f}.pth'.format(
                    epoch, train_loss, val_loss, train_acc, val_ap)
                print('Save as --- ' + file_name)
                save_path = os.path.join(output_dir, file_name)

                torch.save(saver, save_path)
            
            if val_ap == 1:
                break

            #early stopping
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.writer.close()
        dfs_remove_weight(output_dir,retain=3)
        return min_loss, max_acc

    def _train_on_epoch(self, epoch, net, criterion, optimizer, train_loader,scaler):

        net.train()

        train_loss = AverageMeter()
        train_acc = AverageMeter()

        for step, sample in enumerate(train_loader):

            data = sample['image']
            target = sample['label']

            
            data = data.cuda()
            target = target.cuda()
            with autocast(self.use_fp16):
                output = net(data)
                loss = criterion(output, target)
            
            optimizer.zero_grad()
            if self.use_fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            acc = accuracy(output.data, target)[0]
            
            train_loss.update(loss.item(), data.size(0))
            train_acc.update(acc.item(), data.size(0))

            torch.cuda.empty_cache()

            print('epoch:{},step:{},train_loss:{:.5f},train_acc:{:.5f},lr:{}'
                  .format(epoch, step, loss.item(), acc.item(), optimizer.param_groups[0]['lr']))

            if self.global_step % 10 == 0:
                self.writer.add_scalars(
                    'data/train_loss_acc', {'train_loss': loss.item(),
                                            'train_acc': acc.item()}, self.global_step
                )

            self.global_step += 1

        return train_loss.avg, train_acc.avg

    def _val_on_epoch(self, epoch, net, criterion, val_path, label_dict):

        net.eval()

        val_transformer = transforms.Compose(self.transform)

        val_dataset = DataGenerator(
            val_path, label_dict, channels=self.channels, transform=val_transformer)

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        val_loss = AverageMeter()
        val_ap = AverageMeter()

        pred = []
        tar = []

        with torch.no_grad():
            for step, sample in enumerate(val_loader):
                data = sample['image']
                target = sample['label']

                data = data.cuda()
                target = target.cuda()
                with autocast(self.use_fp16):
                    output = net(data)
                    loss = criterion(output, target)

                output = output.float()
                loss = loss.float()

                # measure accuracy and record loss
                acc = accuracy(output.data, target)[0]
                
                val_loss.update(loss.item(), data.size(0))
                val_ap.update(acc.item(), data.size(0))

                pred.extend(output[:,0].detach().tolist())
                tar.extend(target.detach().tolist())

                torch.cuda.empty_cache()

                print('epoch:{},step:{},val_loss:{:.5f},val_acc:{:.5f}'
                      .format(epoch, step, loss.item(), acc.item()))

        pred = np.asarray(pred)
        pred = 1 - pred
        tar = np.asarray(tar)
        
        tar[tar > 0] = 1

        from sklearn.metrics import average_precision_score
        AP = average_precision_score(tar,pred)

        return val_loss.avg, AP

    def _get_net(self, net_name):
        from efficientnet_pytorch import EfficientNet
        if self.external_pretrained:
            net = EfficientNet.from_pretrained(model_name=net_name,
                                            in_channels=self.channels,
                                            num_classes=self.num_classes,
                                            advprop=True)
        else:
            net = EfficientNet.from_name(model_name=net_name)
            num_ftrs = net._fc.in_features
            net._fc = nn.Linear(num_ftrs, self.num_classes)
        return net


    def _get_loss(self, loss_fun, class_weight=None):
        if class_weight is not None:
            class_weight = torch.tensor(class_weight)

        if loss_fun == 'Cross_Entropy':
            loss = nn.CrossEntropyLoss(class_weight)

        return loss

    def _get_optimizer(self, optimizer, net, lr, weight_decay):
        if optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                net.parameters(), 
                lr=lr, weight_decay=weight_decay)

        return optimizer

    def _get_lr_scheduler(self, lr_scheduler, optimizer):
        if lr_scheduler == 'MultiStepLR':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.milestones, gamma=self.gamma)
                
        return lr_scheduler

    def _get_pre_trained(self, weight_path):
        checkpoint = torch.load(weight_path)

        self.net.load_state_dict(checkpoint['state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1


# computing tools

class AverageMeter(object):
    '''
    Computes and stores the average and current value
    '''

    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    '''
    Computes the precision@k for the specified values of k
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1/batch_size))
    return res

class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=True, delta=0, monitor='val_loss',op_type='min'):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement. 
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type

        if self.op_type == 'min':
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):

        score = -val_score if self.op_type == 'min' else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0

    def print_and_update(self, val_score):
        '''print_message when validation score decrease.'''
        if self.verbose:
           print(self.monitor, f'optimized ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...')
        self.val_score_min = val_score