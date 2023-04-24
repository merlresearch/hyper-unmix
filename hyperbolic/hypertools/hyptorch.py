# Copyright (C) 2019 Valentin Khrulkov
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

from hyperbolic.hypertools import pmath


class ToPoincare(nn.Module):

    r"""
    Module which maps points in n-dim Euclidean space
    to n-dim Poincare ball
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None, riemannian=True):
        super(ToPoincare, self).__init__()
        if train_x:
            if ball_dim is None:
                raise ValueError("if train_x=True, ball_dim has to be integer, got {}".format(ball_dim))
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        if train_c:
            self.c = nn.Parameter(
                torch.Tensor(
                    [
                        c,
                    ]
                )
            )
        else:
            self.c = c

        self.train_x = train_x

        self.riemannian = pmath.RiemannianGradient
        self.riemannian.c = c

        if riemannian:
            self.grad_fix = lambda x: self.riemannian.apply(x)
        else:
            self.grad_fix = lambda x: x

    def forward(self, x):

        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return self.grad_fix(pmath.project(pmath.expmap(xp, x, c=self.c), c=self.c))
        return self.grad_fix(pmath.project(pmath.expmap0(x, c=self.c), c=self.c))

    def extra_repr(self):
        return "c={}, train_x={}".format(self.c, self.train_x)
