import os
import yaml
import glob
import torch
from easydict import EasyDict as edict

from PIL import Image

class TestImageFolder(torch.utils.data.Dataset):
  def __init__(self, root, transform=None):
    images = []
    for filename in sorted(glob.glob(os.path.join(root, "*.jpg"))):
      images.append('{}'.format(filename))
    self.root = root
    self.imgs = images
    self.transform = transform

  def __getitem__(self, index):
    filename = self.imgs[index]
    img = Image.open(filename)
    if self.transform is not None:
      img = self.transform(img)
    return img, filename

  def __len__(self):
    return len(self.imgs)

class AverageMeter(object):
  """Computes and stores the average and current value"""
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

def calc_acc(y_pred, y_actual, topk=(1, )):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = y_actual.size(0)

  _, pred = y_pred.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(y_actual.view(1, -1).expand_as(pred))
  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))

  return res


def load_cfg_file(cfg_file):
  with open(cfg_file, 'r') as f:
    yaml_cfg = edict(yaml.load(f))
  return yaml_cfg

def merge_args_to_cfg(cfg, args):
  if cfg is None:
    cfg = edict()
  for k, v in args.__dict__.iteritems():
    if v is not None:
      cfg[k.lower()] = v
  return cfg

def mkdir_if_not_exist(dirs):
  if isinstance(dirs, list):
    for d in dirs:
      mkdir_if_not_exist(d)
  else:
    if not os.path.exists(dirs):
      os.makedirs(dirs)


