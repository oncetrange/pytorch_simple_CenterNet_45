import math
import torch
from torch import nn
import torchvision.ops


class DCN(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1,
               padding=1, dilation=1, deformable_groups=1):
    super().__init__()
    if isinstance(kernel_size, tuple):
      kernel_size = kernel_size[0]
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.deformable_groups = deformable_groups

    self.weight = nn.Parameter(
        torch.empty(out_channels, in_channels, kernel_size, kernel_size))
    self.bias = nn.Parameter(torch.empty(out_channels))
    self.conv_offset_mask = nn.Conv2d(
        in_channels,
        3 * kernel_size * kernel_size * deformable_groups,
        kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
    self._reset_parameters()

  def _reset_parameters(self):
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(self.bias, -bound, bound)
    nn.init.constant_(self.conv_offset_mask.weight, 0.)
    nn.init.constant_(self.conv_offset_mask.bias, 0.)

  def forward(self, x):
    out = self.conv_offset_mask(x)
    o1, o2, mask = torch.chunk(out, 3, dim=1)
    offset = torch.cat([o1, o2], dim=1)
    mask = torch.sigmoid(mask)
    return torchvision.ops.deform_conv2d(
        x, offset, self.weight, self.bias,
        self.stride, self.padding, self.dilation, mask)
