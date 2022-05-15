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
import torch
from torch.nn import Conv2d, Linear, BatchNorm2d, BatchNorm1d, functional as F

from mcs.modules import RMCCell, Identity

from mcs.network.net3d.submodule_3d import SubModule3D

# TODO
class RMC(SubModule3D):
    """
    Relational Memory Core
    https://arxiv.org/pdf/1806.01822.pdf
    """

    def __init__(self, nb_in_chan, output_shape_dict, normalize):
        self.embedding_size = 512
        super(RMC, self).__init__(self.embedding_size, output_shape_dict)
        bias = not normalize
        self.conv1 = Conv2d(
            nb_in_chan, 32, kernel_size=3, stride=2, padding=1, bias=bias
        )
        self.conv2 = Conv2d(
            32, 32, kernel_size=3, stride=2, padding=1, bias=bias
        )
        self.conv3 = Conv2d(
            32, 32, kernel_size=3, stride=2, padding=1, bias=bias
        )
        self.attention = RMCCell(100, 100, 34)
        self.conv4 = Conv2d(
            34, 8, kernel_size=3, stride=1, padding=1, bias=bias
        )
        # BATCH x 8 x 10 x 10
        self.linear = Linear(800, 512, bias=bias)

        if normalize:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(32)
            self.bn3 = BatchNorm2d(32)
            self.bn4 = BatchNorm2d(8)
            self.bn_linear = BatchNorm1d(512)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            self.bn3 = Identity()
            self.bn_linear = Identity()

    def forward(self, input, prev_memories):
        """
        :param input: Tensor{B, C, H, W}
        :param prev_memories: Tuple{B}[Tensor{C}]
        :return:
        """

        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        h = x.size(2)
        w = x.size(3)
        xs_chan = (
            torch.linspace(-1, 1, w)
            .view(1, 1, 1, w)
            .expand(input.size(0), 1, w, w)
            .to(input.device)
        )
        ys_chan = (
            torch.linspace(-1, 1, h)
            .view(1, 1, h, 1)
            .expand(input.size(0), 1, h, h)
            .to(input.device)
        )
        x = torch.cat([x, xs_chan, ys_chan], dim=1)

        # need to transpose because attention expects
        # attention dim before channel dim
        x = x.view(x.size(0), x.size(1), h * w).transpose(1, 2)
        prev_memories = torch.stack(prev_memories)
        x = next_memories = self.attention(x.contiguous(), prev_memories)
        # need to undo the transpose before output
        x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), h, w)

        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_linear(self.linear(x)))
        return x, list(torch.unbind(next_memories, 0))

    @classmethod
    def from_args(cls, args, in_shape, id):
        return cls(in_shape, id, args.fourconv_norm)

    @property
    def _output_shape(self):
        # For 84x84, (32, 5, 5)
        if self._out_shape is None:
            output_dim = calc_output_dim(self._in_shape[1], 3, 2, 1, 1)
            output_dim = calc_output_dim(output_dim, 3, 2, 1, 1)
            output_dim = calc_output_dim(output_dim, 3, 2, 1, 1)
            output_dim = calc_output_dim(output_dim, 3, 1, 1, 1)
            self._out_shape = 32, output_dim, output_dim
        return self._out_shape

    def _forward(self, xs, internals, **kwargs):

        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = F.relu(self.bn2(self.conv2(xs)))
        xs = F.relu(self.bn3(self.conv3(xs)))
        xs = F.relu(self.bn4(self.conv4(xs)))
        return xs, {}

    def _new_internals(self):
        return {}


def calc_output_dim(dim_size, kernel_size, stride, padding, dilation):
    numerator = dim_size + 2 * padding - dilation * (kernel_size - 1) - 1
    return numerator // stride + 1

if __name__ == "__main__":
    output_dim = 84
    output_dim = calc_output_dim(output_dim, 3, 2, 1, 1)
    output_dim = calc_output_dim(output_dim, 3, 2, 1, 1)
    output_dim = calc_output_dim(output_dim, 3, 2, 1, 1)
    output_dim = calc_output_dim(output_dim, 3, 1, 1, 1)
    print(output_dim)  # should be 5

