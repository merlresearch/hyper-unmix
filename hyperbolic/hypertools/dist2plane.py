# Copyright (c) 2023 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2018 Geoopt Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: Apache-2.0

import geoopt.manifolds.stereographic.math as gmath

MIN_NORM = 1e-15


def distance2plane(input, p_k, a_k, ball):

    """
    :param input: batch x space_dim: points (features) in the Poincaré ball
    :return: batch x classes: logit of probabilities for 'out_features' classes
    """

    def _dist2plane(x, a, p, c, k, keepdim: bool = False, signed: bool = False, dim: int = -1):
        """
        Reimplementation from geoopt and corrected so it returns a_norm
        and this value does not have to be calculated twice
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

    input = input.unsqueeze(-2)  # batch x aux x space_dim
    distance, a_norm = _dist2plane(x=input, p=p_k, a=a_k, c=ball.c, k=ball.k, signed=True)
    result = 2 * a_norm * distance
    return result
