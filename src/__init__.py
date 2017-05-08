from basic_model import BasicModel
from convnet import ConvNet, DNDF
from convnet import model_names as convnet_names
from utils import load_cfg_file, merge_args_to_cfg, mkdir_if_not_exist, TestImageFolder


def make_model(cfg):
  if cfg['model_name'] in convnet_names:
    return ConvNet(cfg)
  elif cfg['model_name'] == 'dndf':
    return DNDF(cfg)
  else:
    raise NotImplementedError