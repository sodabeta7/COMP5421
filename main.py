import os
import sys
import argparse
import os.path as osp

from src import make_model
from src import convnet_names
from src import merge_args_to_cfg, load_cfg_file, mkdir_if_not_exist

parser = argparse.ArgumentParser(description='Pytorch Binary Image Classification')
parser.add_argument('--cfg', help='config file')
parser.add_argument('--data_dir', default='data', help='data directory')
parser.add_argument('--result_dir', default='results', help='result directory')

parser.add_argument('--exp_name', required=True, help='name of experiment enviroment')
parser.add_argument('--model_name', help='name of model')
parser.add_argument('--max_epochs', type=int, default=500, help='number of maximum epochs for training')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--optimizer', default='sgd', help='optimizer algorithm')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--resume', help='checkpoint file to load')
parser.add_argument('--mode', default='train', help='running mode: train, val, test')
parser.add_argument('--csv_name', default='test', help='name of the csv file')

def main():
  args = parser.parse_args()
  cfg = None
  if args.cfg:
    cfg = load_cfg_file(args.cfg)
  cfg = merge_args_to_cfg(cfg, args)
  cfg.result_dir = osp.join(cfg.result_dir, cfg.exp_name)
  for phase in ['train', 'val', 'test']:
    cfg['data_%s_dir'%phase] = osp.join(cfg.data_dir, phase)
  cfg['ckpt_dir'] = osp.join(cfg.result_dir, 'ckpts')
  cfg['log_dir'] = osp.join(cfg.result_dir, 'logs')
  cfg['eval_res_dir'] = osp.join(cfg.result_dir, 'eval_res')
  mkdir_if_not_exist([cfg.result_dir, cfg.ckpt_dir, cfg.log_dir, cfg.eval_res_dir])

  print('using config')
  import pprint
  pprint.pprint(cfg)

  model = make_model(cfg)

  if cfg.mode == 'train':
    model.train()
  elif cfg.mode == 'val':
    model.validate()
  else:
    model.test()


main()
