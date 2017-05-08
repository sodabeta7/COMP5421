import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from src.basic_model import BasicModel


model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))


# ['alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']

class ConvNet(BasicModel):
  def build_model(self, cfg):
    model_name = cfg['model_name']
    assert model_name in model_names
    if model_name.endswith('_bn'):
      self.model = models.__dict__[model_name]()
    else:  
      self.model = models.__dict__[model_name](pretrained=cfg['use_pretrained_params'])
    self.criterion = nn.CrossEntropyLoss().cuda()
    self.optimizer = self.create_optimizer(cfg)

    #fine-tuning model
    # for param in self.model.parameters():
      # param.requires_grad = False
    # import pdb
    # pdb.set_trace()
    if model_name.startswith('resnet'):
      num_ftrs = self.model.fc.in_features
      self.model.fc = nn.Linear(num_ftrs, cfg['num_classes'])
    elif model_name.startswith('dense'):
      num_ftrs = self.model.classifier.in_features
    elif model_name.startswith('alex'):
      fc = self.model.classifier._modules[str(len(self.model.classifier._modules) - 1)]
      num_ftrs = fc.in_features
      self.model.classifier._modules[str(len(self.model.classifier._modules) - 1)] = nn.Linear(num_ftrs, cfg['num_classes'])
    elif model_name.startswith('inception'):
      # import pdb
      # pdb.set_trace()
      num_ftrs = self.model.fc.in_features
      self.model.fc = nn.Linear(num_ftrs, cfg['num_classes'])
      num_ftrs = self.model.AuxLogits.fc.in_features
      self.model.AuxLogits.fc = nn.Linear(num_ftrs, cfg['num_classes'])
    if model_name.startswith('alexnet') or model_name.startswith('vgg'):
      self.model.features = torch.nn.DataParallel(self.model.features).cuda()
      self.model.cuda()
    else:
      self.model = torch.nn.DataParallel(self.model).cuda()


  @property
  def trainable_parameters(self):
    # return self.model.fc.parameters()
    return self.model.parameters()

class DNDF(BasicModel):
  def build_model(self, cfg):
    self.model = DeepNeuralDecisionForest(cfg).cuda()
    self.criterion = nn.CrossEntropyLoss().cuda()
    self.optimizer = self.create_optimizer(cfg)

  @property
  def trainable_parameters(self):
    return self.model.parameters()


class DeepNeuralDecisionForest(nn.Module):
    def __init__(self, cfg):
        super(DeepNeuralDecisionForest, self).__init__()

        self.conv = models.__dict__['resnet18'](pretrained=cfg['use_pretrained_params'])
        self.num_ftrs = self.conv.fc.in_features
        self.conv = torch.nn.Sequential(*list(self.conv.children())[:-1])
        self.conv = torch.nn.DataParallel(self.conv).cuda()

        p_keep_hidden = cfg['p_keep_hidden']
        self._nlabel = cfg['num_classes']
        self._ntree = cfg['n_tree']
        self._ndepth = cfg['n_depth']
        self._nleaf = 2 ** (self._ndepth + 1)
        self._batchsize = cfg['batch_size']

        self.treelayers = nn.ModuleList()
        self.pi_e = nn.ParameterList()
        for i in range(self._ntree):
            treelayer = nn.Sequential()
            treelayer.add_module('sub_linear1', nn.Linear(self.num_ftrs, 128))
            treelayer.add_module('sub_relu', nn.ReLU())
            treelayer.add_module('sub_drop1', nn.Dropout(1-p_keep_hidden))
            treelayer.add_module('sub_linear2', nn.Linear(128, self._nleaf))
            treelayer.add_module('sub_sigmoid', nn.Sigmoid())
           
            self.treelayers.append(treelayer)
            self.pi_e.append(Parameter(self.init_prob_weights([self._nleaf, self._nlabel], -2, 2)))

    def init_pi(self):
        return torch.ones(self._nleaf, self._nlabel)/float(self._nlabel)

    def init_weights(self, shape):
        return torch.randn(shape).uniform(-0.01,0.01)

    def init_prob_weights(self, shape, minval=-5, maxval=5):
        return torch.Tensor(shape[0], shape[1]).uniform_(minval, maxval)

    def compute_mu(self, flat_decision_p_e):
        n_batch = self._batchsize
        batch_0_indices = torch.range(0, n_batch * self._nleaf - 1, self._nleaf).unsqueeze(1).repeat(1, self._nleaf).long()

        in_repeat = self._nleaf // 2
        out_repeat = n_batch

        batch_complement_indices = torch.LongTensor(
            np.array([[0] * in_repeat, [n_batch * self._nleaf] * in_repeat] * out_repeat).reshape(n_batch, self._nleaf))

        # First define the routing probabilistics d for root nodes
        mu_e = []
        indices_var = Variable((batch_0_indices + batch_complement_indices).view(-1)) 
        indices_var = indices_var.cuda()
        # iterate over each tree
        for i, flat_decision_p in enumerate(flat_decision_p_e):
            mu = torch.gather(flat_decision_p, 0, indices_var).view(n_batch, self._nleaf)
            mu_e.append(mu)

        # from the scond layer to the last layer, we make the decison nodes
        for d in range(1, self._ndepth + 1):
            indices = torch.range(2 ** d, 2 ** (d + 1) - 1) - 1
            tile_indices = indices.unsqueeze(1).repeat(1, 2 ** (self._ndepth - d + 1)).view(1, -1)
            batch_indices = batch_0_indices + tile_indices.repeat(n_batch, 1).long()

            in_repeat = in_repeat // 2
            out_repeat = out_repeat * 2
            # Again define the indices that picks d and 1-d for the nodes
            batch_complement_indices = torch.LongTensor(
                np.array([[0] * in_repeat, [n_batch * self._nleaf] * in_repeat] * out_repeat).reshape(n_batch, self._nleaf))

            mu_e_update = []
            indices_var = Variable((batch_indices + batch_complement_indices).view(-1))
            indices_var = indices_var.cuda()
            for mu, flat_decision_p in zip(mu_e, flat_decision_p_e):
                mu = torch.mul(mu, torch.gather(flat_decision_p, 0, indices_var).view(
                    n_batch, self._nleaf))
                mu_e_update.append(mu)
            mu_e = mu_e_update
        return mu_e

    def compute_py_x(self, mu_e, leaf_p_e):
        py_x_e = []
        n_batch = self._batchsize

        for i in range(len(mu_e)):
            py_x_tree = mu_e[i].unsqueeze(2).repeat(1, 1, self._nlabel).mul(leaf_p_e[i].unsqueeze(0).repeat(n_batch, 1, 1)).mean(1)
            py_x_e.append(py_x_tree.squeeze().unsqueeze(0))

        py_x_e = torch.cat(py_x_e, 0)
        py_x = py_x_e.mean(0).squeeze()
        
        return py_x

    def forward(self, x):
        feat = self.conv.forward(x)
        feat = feat.view(-1, self.num_ftrs)
        self._batchsize = x.size(0)

        flat_decision_p_e = []
        leaf_p_e = []
        
        for i in range(len(self.treelayers)):
            decision_p = self.treelayers[i].forward(feat)
            decision_p_comp = 1 - decision_p
            decision_p_pack = torch.cat((decision_p, decision_p_comp))
            flat_decision_p = decision_p_pack.view(-1)
            flat_decision_p_e.append(flat_decision_p)
            leaf_p = F.softmax(self.pi_e[i])
            leaf_p_e.append(leaf_p)
        
        mu_e = self.compute_mu(flat_decision_p_e)
        
        py_x = self.compute_py_x(mu_e, leaf_p_e)
        return torch.log(py_x)