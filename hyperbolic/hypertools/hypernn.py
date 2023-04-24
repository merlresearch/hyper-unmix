# Copyright (c) 2023 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2020 HITS NLP
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

"""
Network definitions adapted from https://github.com/nlpAThits/hyfi/blob/master/hyfi/hypernn.py
"""

import geoopt
import geoopt.manifolds.stereographic.math as gmath
import numpy as np
import torch

MIN_NORM = 1e-15


def certainty_from_coors(coors):
    return np.linalg.norm(coors, axis=-1)


class MobiusMLR(torch.nn.Module):
    """
    Multinomial logistic regression in the Poincare Ball
    It is based on formulating logits as distances to margin hyperplanes.
    In Euclidean space, hyperplanes can be specified with a point of origin
    and a normal vector. The analogous notion in hyperbolic space for a
    point $p \in \mathbb{D}^n$ and
    $a \in T_{p} \mathbb{D}^n \backslash \{0\}$ would be the union of all
    geodesics passing through $p$ and orthogonal to $a$. Given $K$ classes
    and $k \in \{1,...,K\}$, $p_k \in \mathbb{D}^n$,
    $a_k \in T_{p_k} \mathbb{D}^n \backslash \{0\}$, the formula for the
    hyperbolic MLR is:
    \begin{equation}
        p(y=k|x) f\left(\lambda_{p_k} \|a_k\| \operatorname{sinh}^{-1} \left(\frac{2 \langle -p_k \oplus x, a_k\rangle}
                {(1 - \| -p_k \oplus x \|^2)\|a_k\|} \right) \right)
    \end{equation}
    """

    def __init__(self, in_features, out_features, c=1.0):
        """
        :param in_features: number of dimensions of the input
        :param out_features: number of classes
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = geoopt.PoincareBall(c=c)
        points = torch.randn(out_features, in_features) * 1e-5
        points = gmath.expmap0(points, k=self.ball.k)
        self.p_k = geoopt.ManifoldParameter(points, manifold=self.ball)

        tangent = torch.Tensor(out_features, in_features)
        stdv = (6 / (out_features + in_features)) ** 0.5  # xavier uniform
        torch.nn.init.uniform_(tangent, -stdv, stdv)
        self.a_k = torch.nn.Parameter(tangent)

    def forward(self, input):
        """
        :param input: batch x space_dim: points (features) in the Poincaré ball
        :return: batch x classes: logit of probabilities for 'out_features' classes
        """
        input = input.unsqueeze(-2)  # batch x aux x space_dim
        distance, a_norm = self._dist2plane(x=input, p=self.p_k, a=self.a_k, c=self.ball.c, k=self.ball.k, signed=True)
        result = 2 * a_norm * distance
        return result

    def _dist2plane(self, x, a, p, c, k, keepdim: bool = False, signed: bool = False, dim: int = -1):
        """
        Taken from geoopt and corrected so it returns a_norm and this value does not have to be calculated twice
        """
        sqrt_c = c**0.5
        minus_p_plus_x = gmath.mobius_add(-p, x, k=k, dim=dim)
        mpx_sqnorm = minus_p_plus_x.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(MIN_NORM)
        mpx_dot_a = (minus_p_plus_x * a).sum(dim=dim, keepdim=keepdim)
        if not signed:
            mpx_dot_a = mpx_dot_a.abs()
        a_norm = a.norm(dim=dim, keepdim=keepdim, p=2).clamp_min(MIN_NORM)
        num = 2 * sqrt_c * mpx_dot_a
        denom = (1 - c * mpx_sqnorm) * a_norm
        return gmath.arsinh(num / denom.clamp_min(MIN_NORM)) / sqrt_c, a_norm

    def extra_repr(self):
        return "in_features={in_features}, out_features={out_features}".format(**self.__dict__) + f" k={self.ball.k}"


class EuclMLR(torch.nn.Module):
    """Euclidean Multinomial logistic regression"""

    def __init__(self, in_features, out_features):
        """
        :param in_features: number of dimensions of the input
        :param out_features: number of classes
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = geoopt.Euclidean()
        points = torch.randn(out_features, in_features) * 1e-5
        self.p_k = geoopt.ManifoldParameter(points, manifold=self.manifold)

        tangent = torch.Tensor(out_features, in_features)
        stdv = (6 / (out_features + in_features)) ** 0.5  # xavier uniform
        torch.nn.init.uniform_(tangent, -stdv, stdv)
        self.a_k = torch.nn.Parameter(tangent)

    def forward(self, input):
        """
        :param input: batch x space_dim: points (features) in the Poincaré ball
        :return: batch x classes: logit of probabilities for 'out_features' classes
        """
        x = input.unsqueeze(dim=-2)
        minus_p_plus_x = -self.p_k + x
        result = (minus_p_plus_x * self.a_k).sum(dim=-1, keepdim=False)
        return 4 * result

    def extra_repr(self):
        return "in_features={in_features}, out_features={out_features}".format(**self.__dict__)
