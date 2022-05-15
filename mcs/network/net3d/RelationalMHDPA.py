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
from mcs.network.net2d.submodule_2d import SubModule2D
import torch.nn as nn
import torch
import numpy as np
import math


class RelationalMHDPA(nn.Module):
    args = {}

    def __init__(self, input_shape, nb_head, scale=False):
        super(RelationalMHDPA, self).__init__()
        self._out_shape = input_shape


        nb_channel = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]

        assert nb_channel % nb_head == 0
        seq_len = height * width
        self.register_buffer(
            "b",
            torch.tril(torch.ones(seq_len, seq_len)).view(
                1, 1, seq_len, seq_len
            ),
        )
        self.nb_head = nb_head
        self.split_size = nb_channel
        self.scale = scale
        self.projection = nn.Linear(nb_channel, nb_channel * 3)
        self.re = nn.ReLU()
        self.mlp = nn.Linear(nb_channel, nb_channel)

    def forward(self, x):
        """
              :param x: A tensor with a shape of [batch, seq_len, nb_channel]
              :return: A tensor with a shape of [batch, seq_len, nb_channel]
              """

        size_out = x.size()[:-1] + (self.split_size * 3,)  # [batch, seq_len, nb_channel*3]

        x = self.projection(x.view(-1, x.size(-1)))  # [BT,C]
        # x = self.re(x)
        x = x.view(*size_out)  # [B,T,3C]

        query, key, value = x.split(self.split_size, dim=2) 
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        a = self._attn(query, key, value)
        e = self.merge_heads(a)

        return self.mlp(e)

    def _new_internals(self):
        return {}

    def _attn(self, q, k, v):  
        w = torch.matmul(q, k)

        if self.scale:
            w = w / math.sqrt(v.size(-1))
        w = w * self.b + -1e9 * (
                1 - self.b
        )  # TF implem method: mask_attn_weights
        w = nn.Softmax(dim=-1)(w)
        h = torch.matmul(w, v)

        return h

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        # keep dims, but expand the last dim to be [head, chan // head] X[B,T,C]
        new_x_shape = x.size()[:-1] + (self.nb_head, x.size(-1) // self.nb_head)  # [B,T,H,C//H]
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            # batch, head, channel, attend
            return x.permute(0, 2, 3, 1)
        else:
            # batch, head, attend, channel
            return x.permute(0, 2, 1, 3)

    def get_parameter_names(self, layer):
        return [
            "Proj{}_W".format(layer),
            "Proj{}_b".format(layer),
            "MLP{}_W".format(layer),
            "MLP{}_b".format(layer),
        ]
