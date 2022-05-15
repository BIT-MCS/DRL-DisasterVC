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

import torch.nn as nn
import torch
from ..net2d.submodule_2d import SubModule2D


class DeConv2D(SubModule2D):
    args = {}

    def __init__(self, input_shape, id):
        super().__init__(input_shape, id)
        self._input_shape = input_shape
        self._out_shape = None
        self._pc_layers = 16
        self.num_action = 6

        self.fc = nn.Sequential(
            nn.Linear(in_features=input_shape[0], out_features=9 * 9 * self._pc_layers), nn.ReLU())

        self.deconv_v = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self._pc_layers, out_channels=1, kernel_size=5, stride=2),
            nn.ReLU())
        self.deconv_a = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self._pc_layers, out_channels=self.num_action, kernel_size=5, stride=2),
            nn.ReLU())

    @classmethod
    def from_args(cls,args,input_shape, id):
        return cls(input_shape, id)

    @property
    def _output_shape(self):
        if self._out_shape is None:
            self._out_shape = 21*21,self.num_action
        return self._out_shape

    def _forward(self, input, internals=None, **kwargs):
        #print(input.shape) # 32 512 1
        input = self.fc(input.view(-1, 512))
        input = input.view([-1, self._pc_layers, 9, 9])
        v = self.deconv_v(input)
        a = self.deconv_a(input)
        a_mean = torch.mean(a, dim=1, keepdim=True)
        q = v + a - a_mean  
        out = q.reshape(-1, self.num_action, 21 * 21)
        #print(out.shape) # 32 6 441
        return out.permute(0, 2, 1).contiguous(),{}

    def _new_internals(self):
        return {}

    @_output_shape.setter
    def _output_shape(self, value):
        self.__output_shape = value



def calc_output_dim(dim_size, kernel_size, stride=1, input_padding=0, output_padding=0, dilation=1):
    numerator = (dim_size - 1) * stride - 2 * input_padding + dilation * (kernel_size - 1) + output_padding + 1
    return numerator


if __name__ == "__main__":
    out = [4, 5, 1]
    for output_dim in out:
        output_dim = calc_output_dim(output_dim, 5, 2)
        print(output_dim)
