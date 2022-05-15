# Copyright (C) 2018 Heron Systems, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from torch.nn import Conv2d, BatchNorm2d, GroupNorm, init, functional as F

from mcs.modules import Identity
from mcs.network.net3d.submodule_3d import SubModule3D
from mcs.network.net3d.RelationalMHDPA import RelationalMHDPA


class FourConv(SubModule3D):
    args = {"fourconv_norm": "bn"}

    def __init__(self, in_shape, id, normalize, args):
        super().__init__(in_shape, id)
        bias = not normalize
        self._in_shape = in_shape
        self._out_shape = None
        self._args = args
        self.conv1 = Conv2d(in_shape[0], 32, 7, stride=2, padding=1, bias=bias)
        self.conv2 = Conv2d(32, 32, 3, stride=2, padding=1, bias=bias)
        self.conv3 = Conv2d(32, 32, 3, stride=2, padding=1, bias=bias)
        self.conv4 = Conv2d(32, 32, 3, stride=2, padding=1, bias=bias)

        self.use_mhra = args.use_mhra
        if normalize == "bn":
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(32)
            self.bn3 = BatchNorm2d(32)
            self.bn4 = BatchNorm2d(32)
        elif normalize == "gn":
            self.bn1 = GroupNorm(8, 32)
            self.bn2 = GroupNorm(8, 32)
            self.bn3 = GroupNorm(8, 32)
            self.bn4 = GroupNorm(8, 32)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            self.bn3 = Identity()
            self.bn4 = Identity()
        if args.use_mhra:
            self.att = RelationalMHDPA(input_shape=(32, 5, 5), nb_head=args.num_head)

        relu_gain = init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

    @classmethod
    def from_args(cls, args, in_shape, id):
        return cls(in_shape, id, args.linear_normalize,args)

    @property
    def _output_shape(self):
        # For 84x84, (32, 5, 5)
        if self._out_shape is None:
            output_dim = calc_output_dim(self._in_shape[1], 7, 2, 1, 1)
            output_dim = calc_output_dim(output_dim, 3, 2, 1, 1)
            output_dim = calc_output_dim(output_dim, 3, 2, 1, 1)
            output_dim = calc_output_dim(output_dim, 3, 2, 1, 1)
            self._out_shape = 32, output_dim, output_dim
        return self._out_shape

    def _forward(self, xs, internals, **kwargs):

        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = F.relu(self.bn2(self.conv2(xs)))
        xs = F.relu(self.bn3(self.conv3(xs)))
        xs = F.relu(self.bn4(self.conv4(xs)))
        if self.use_mhra:
            xs =xs + self._atten(xs, self.att)
        return xs, {}

    def _new_internals(self):
        return {}

    def _atten(self, x, att):
        W, H = x.shape[-2:]
        h = x.view(-1, 32, W * H)
        h = h.permute(0, 2, 1).contiguous()
        h = att(h)
        h = h.permute(0, 2, 1).contiguous()
        h = h.view(-1, 32, W, H)
        return h


def calc_output_dim(dim_size, kernel_size, stride, padding, dilation):
    numerator = dim_size + 2 * padding - dilation * (kernel_size - 1) - 1
    return numerator // stride + 1


if __name__ == "__main__":
    output_dim = 84
    output_dim = calc_output_dim(output_dim, 7, 2, 1, 1)
    output_dim = calc_output_dim(output_dim, 3, 2, 1, 1)
    output_dim = calc_output_dim(output_dim, 3, 2, 1, 1)
    print(output_dim)
    output_dim = calc_output_dim(output_dim, 3, 2, 1, 1)
    print(output_dim)  # should be 5
