# local and global alignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from fvcore.nn.focal_loss import sigmoid_focal_loss_jit
from .build import DA_HEAD_REGISTRY
from ..layers.gradient_scalar_layer import GradientScalarLayer

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class GlobalAlignmentHead(nn.Module):
  output_channel = 256
  def __init__(self, context=False):
    super(GlobalAlignmentHead, self).__init__()
    self.conv1 = conv3x3(1024, 512, stride=2)
    self.bn1 = nn.BatchNorm2d(512)
    self.conv2 = conv3x3(512, self.output_channel, stride=2)
    self.bn2 = nn.BatchNorm2d(self.output_channel)
    self.conv3 = conv3x3(self.output_channel, self.output_channel, stride=2)
    self.bn3 = nn.BatchNorm2d(self.output_channel)
    self.fc = nn.Linear(self.output_channel, 2)
    self.context = context
    self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

  def forward(self, x):
    x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
    x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
    x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
    x = F.avg_pool2d(x, (x.size(2),x.size(3)))
    x = x.view(-1, self.output_channel)
    if self.context:
      feat = x
    x = self.fc(x)
    if self.context:
      return x, feat
    else:
      return x


class LocalAlignmentHead(nn.Module):
  def __init__(self, context=False):
    super(LocalAlignmentHead, self).__init__()
    self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
    self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
    self.conv3 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=False)
    self.context = context
    self._init_weights()

  def _init_weights(self):
    def normal_init(m, mean, stddev):
      """
      weight initalizer: random normal.
      """
      # x is a parameter
      m.weight.data.normal_(mean, stddev)
      # m.bias.data.zero_()

    normal_init(self.conv1, 0, 0.01)
    normal_init(self.conv2, 0, 0.01)
    normal_init(self.conv3, 0, 0.01)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    if self.context:
      feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
      x = self.conv3(x)
      return x, feat
    else:
      x = self.conv3(x)
      return x


@DA_HEAD_REGISTRY.register()
class AlignmentHead(nn.Module):

  @configurable
  def __init__(self, *, gamma=5.0):
    super().__init__()
    self.localhead = LocalAlignmentHead(context=True)
    self.grl_localhead = GradientScalarLayer(-1.0)
    self.globalhead = GlobalAlignmentHead(context=True)
    self.grl_globalhead = GradientScalarLayer(-1.0)
    self.gamma = gamma

  @classmethod
  def from_config(cls, cfg):
    return {'gamma': cfg.DA_HEAD.GAMMA}
  
  def forward(self, inputs):
    '''
    inputs: 
      dict[str->Tensor], local feature, global feature, feature domain
    returns:

    '''
    feat_local = inputs['local feature']
    feat_global = inputs['global feature']
    feat_domain = inputs['feature domain']
    # localhead branch
    _, reg_local_feat = self.localhead(feat_local.detach())
    # globalhead branch
    _, reg_global_feat = self.globalhead(feat_global.detach())

    if self.training:
      feat_2d, _ = self.localhead(self.grl_localhead(feat_local))
      feat_value, _ = self.globalhead(self.grl_globalhead(feat_global))
      if feat_domain == 'source':
        domain_label = torch.ones_like(feat_value, requires_grad=True, device=feat_2d.device)
        # local alignment, gan loss, l2-norm
        domain_loss_local = 0.5 * torch.mean(torch.sigmoid(feat_2d) ** 2)
      elif feat_domain == 'target':
        domain_label = torch.zeros_like(feat_value, requires_grad=True, device=feat_2d.device)
        # local alignment, gan loss, l2-norm
        domain_loss_local = 0.5 * torch.mean(torch.sigmoid(1 - feat_2d) ** 2)

      # global alignment, focal loss  
      focal_loss_global = sigmoid_focal_loss_jit(feat_value, domain_label, gamma=self.gamma, reduction='mean')
      return reg_local_feat, reg_global_feat, {'local_alignment_loss': domain_loss_local, 'global_alignment_loss': focal_loss_global}
    else:
      return reg_local_feat, reg_global_feat


def build_da_heads(cfg):
    return AlignmentHead(cfg)
