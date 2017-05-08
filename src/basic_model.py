import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import os.path as osp
import os, csv, copy, time, shutil, collections
from torchvision import datasets, transforms

from utils import TestImageFolder, AverageMeter, calc_acc

class BasicModel(object):
  """Basic model class"""
  def __init__(self, cfg):
    self.cfg = cfg

    self.build_model(cfg)
    self.init(cfg)

  def build_model(self, cfg):
    pass

  @property
  def trainable_parameters(self):
    pass

  def create_optimizer(self, cfg):
    if cfg['optimizer'] == 'sgd':
      return torch.optim.SGD(
              self.trainable_parameters,
              lr=cfg['lr'],
              momentum=cfg['momentum'],
              weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'adam':
      return torch.optim.Adam(self.trainable_parameters,
                              lr=cfg['lr'],
                              weight_decay=cfg['weight_decay'])
    else:
      raise NotImplementedError

  def init(self, cfg):
    train_dir = cfg['data_train_dir']
    val_dir = cfg['data_val_dir']
    test_dir = cfg['data_test_dir']

    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    scale_size = 299 if cfg['model_name'] == 'inception_v3' else 256
    in_size = 299 if cfg['model_name'] == 'inception_v3' else 224

    self.train_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(train_dir, transforms.Compose([
        transforms.RandomSizedCrop(in_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])),
      batch_size=cfg['batch_size'],
      shuffle=True,
      num_workers=cfg['data_loader_workers'],
      pin_memory=True)

    self.val_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(val_dir, transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(in_size),
        transforms.ToTensor(),
        normalize])),
      batch_size=cfg['batch_size'],
      shuffle=False,
      num_workers=cfg['data_loader_workers'],
      pin_memory=True)

    self.test_loader = torch.utils.data.DataLoader(
      TestImageFolder(test_dir, transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(in_size),
        transforms.ToTensor(),
        normalize])),
      batch_size=cfg['batch_size'],
      shuffle=False,
      num_workers=cfg['data_loader_workers'],
      pin_memory=True)

    self.best_acc = 0
    self.start_epoch = 0
    if cfg['resume']:
      self.load_ckpt(cfg['resume'])

  def train_epoch(self, loader, epoch):
    losses = AverageMeter()
    acc = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    self.model.train()

    end = time.time()
    for batch_step, (images, target) in enumerate(loader):
      data_time.update(time.time() - end)
      target = target.cuda(async=True)
      image_var = Variable(images)
      label_var = Variable(target)

      y_pred = self.model(image_var)
      # import pdb
      # pdb.set_trace()
      if isinstance(y_pred, tuple):
        loss_lab = self.criterion(y_pred[0], label_var)
        loss_aux = self.criterion(y_pred[1], label_var)
        loss = loss_lab + loss_aux
        y_pred = y_pred[0]
      else:
        loss = self.criterion(y_pred, label_var)

      batch_acc, _ = calc_acc(y_pred.data, target, topk=(1, 1))
      losses.update(loss.data[0], images.size(0))
      acc.update(batch_acc[0], images.size(0))

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      batch_time.update(time.time() - end)
      end = time.time()
      if batch_step % self.cfg['print_freq'] == 0:
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, batch_step, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=acc))

  def validate(self):
    losses = AverageMeter()
    acc = AverageMeter()

    self.model.eval()

    loader = self.val_loader
    for batch_step, (images, target) in enumerate(loader):
      target = target.cuda(async=True)
      image_var = Variable(images, volatile=True)
      label_var = Variable(target, volatile=True)
      y_pred = self.model(image_var)
      loss = self.criterion(y_pred, label_var)

      batch_acc, _ = calc_acc(y_pred.data, target, topk=(1, 1))
      losses.update(loss.data[0], images.size(0))
      acc.update(batch_acc[0], images.size(0))

      if batch_step % self.cfg['print_freq'] == 0:
        print('Val: [{0}/{1}]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                batch_step, len(loader), loss=losses, top1=acc))
    print('Val Accuracy %.6f | Loss %.6f' % (acc.avg, losses.avg))
    return acc.avg

  def test(self):
    csv_map = collections.defaultdict(float)
    self.model.eval()
    
    print("Evaluation on test set")
    for i, (images, filepaths) in enumerate(self.test_loader):
      image_var = torch.autograd.Variable(images, volatile=True)
      smax = torch.nn.Softmax()
      y_pred = self.model(image_var)
      y_pred = smax(y_pred).data.cpu().numpy()
      # import pdb
      # pdb.set_trace()
      for j in xrange(images.size(0)):
        # y = y_pred[j].argmax()
        smax_out = y_pred[j]
        cat_prob = smax_out[0]
        dog_prob = smax_out[1]
        prob = dog_prob
        if cat_prob > dog_prob:
            prob = 1 - cat_prob
        prob = np.around(prob, decimals=4)
        prob = np.clip(prob, .0001, .999)

        # pop extension, treat as id to map
        filepath = filepaths[j]
        filepath = os.path.splitext(os.path.basename(filepath))[0]
        filepath = int(filepath)
        csv_map[filepath] = prob

    sub_fn = osp.join(self.cfg['eval_res_dir'], self.cfg['csv_name'] + '.csv')
    print("Writing Predictions to CSV...")
    with open(sub_fn, 'w') as csvfile:
        fieldnames = ['id', 'label']
        csv_w = csv.writer(csvfile)
        csv_w.writerow(('id', 'label'))
        for row in sorted(csv_map.items()):
            csv_w.writerow(row)
    print("Done.")

  def train(self):
    self.model.train()

    acc_window = []
    for epoch in xrange(self.start_epoch, self.cfg['max_epochs']):
      self.train_epoch(self.train_loader, epoch)
      acc = self.validate()

      min_old_acc = 0 if len(acc_window) == 0 else min(acc_window)
      if acc <= min_old_acc:
        self.adjust_lr()
      acc_window.append(acc)
      if len(acc_window) > 2:
        acc_window = acc_window[1:]

      is_best = acc > self.best_acc
      self.best_acc = max(self.best_acc, acc)
      self.save_ckpt({
        'epoch': epoch + 1,
        'cfg': self.cfg,
        'state_dict': self.model.state_dict(),
        'best_acc': self.best_acc,
        'optimizer': self.optimizer.state_dict()
      }, is_best, 'Epoch-{epoch}-ckpt.pth.bar'.format(epoch=epoch + 1))

  def adjust_lr(self):
    lr = self.cfg['lr'] / 10.0
    print('===>Adjusting learning rate to %.6f' % lr)
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr

  def save_ckpt(self, state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, osp.join(self.cfg['ckpt_dir'], filename))
    if is_best:
      shutil.copyfile(osp.join(self.cfg['ckpt_dir'], filename),
                      osp.join(self.cfg['ckpt_dir'], 'best_model.pth.bar'))

  def load_ckpt(self, ckpt_file):
    if osp.isfile(ckpt_file):
      print("=> loading checkpoint '{}'".format(ckpt_file))
      checkpoint = torch.load(ckpt_file)
      self.start_epoch = checkpoint['epoch']
      self.best_acc = checkpoint['best_acc']
      self.model.load_state_dict(checkpoint['state_dict'])
      self.optimizer.load_state_dict(checkpoint['optimizer'])
      print("=> loaded checkpoint '{}' (epoch {})".format(ckpt_file, checkpoint['epoch']))
    else:
      raise ValueError('checkpoint file does not exist')