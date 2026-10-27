import numpy as np
import torch

from calcdiff.expr import Add, Const, Div, Fn, Mul, Pow, Var
from calcdiff.generate import Sampler
from calcdiff.models.tree_net import TreeBT, TreeEncoder, compile_forest


def naive(enc: TreeEncoder, node) -> torch.Tensor:
    """Plain recursive evaluation of the same network, node by node."""
    match node:
        case Var():
            return enc.x
        case Const(v):
            return enc.const(torch.tensor([np.log1p(abs(float(v))), np.sign(float(v))], dtype=torch.float32))
        case Pow(b, e):
            extra = torch.tensor([float(e), float(e < 0), float(e.denominator != 1)])
            return torch.tanh(enc.pow(torch.cat([naive(enc, b), extra])))
        case Add(ts):
            return torch.tanh(enc.add(sum(naive(enc, t) for t in ts)))
        case Mul(fs):
            return torch.tanh(enc.mul(sum(naive(enc, f) for f in fs)))
        case Div(n, d):
            return torch.tanh(enc.div(torch.cat([naive(enc, n), naive(enc, d)])))
        case Fn(g, a):
            return torch.tanh(enc.fn[g](naive(enc, a)))


def test_batched_forest_equals_naive_recursion():
    torch.manual_seed(0)
    enc = TreeEncoder(8)
    trees = [p.tree for p in Sampler(seed=3).sample_unique(60)]
    batched = enc(compile_forest(trees))
    for k, t in enumerate(trees):
        assert torch.allclose(batched[k], naive(enc, t), atol=1e-5)


def test_sum_is_order_free_but_quotient_is_not():
    from fractions import Fraction as Q
    from calcdiff.expr import X
    enc = TreeEncoder(8)
    a, b = Pow(X, Q(2)), Fn("sin", X)
    H = enc(compile_forest([Add((a, b)), Add((b, a)), Div(a, b), Div(b, a)]))
    assert torch.allclose(H[0], H[1]) and not torch.allclose(H[2], H[3])


def test_fits_a_learnable_signal():
    from calcdiff.features import feature_matrix
    from calcdiff.synthetic import random_pairs, simulate_judgments
    rng = np.random.default_rng(0)
    probs = Sampler(seed=4).sample_unique(200)
    Phi = feature_matrix(p.tree for p in probs)
    delta = 0.15 * Phi[:, 3] - 1.5                      # bigger derivative = harder
    pairs = random_pairs(200, 3000, rng)
    y = simulate_judgments(delta, pairs, rng)
    model = TreeBT([p.tree for p in probs], dim=8, max_epochs=300).fit(pairs, y)
    s = model.score()
    from scipy.stats import spearmanr
    assert spearmanr(s, delta)[0] > 0.8


def test_lp_ft_restores_freeze_setting_and_keeps_pretrained_signal():
    from calcdiff.features import feature_matrix
    from calcdiff.models.tree_net import pretrain_encoder
    from calcdiff.synthetic import random_pairs, simulate_judgments
    gen = [p.tree for p in Sampler(seed=8).sample_unique(800)]
    enc, losses = pretrain_encoder(gen, feature_matrix(gen), dim=8, epochs=150)
    assert losses[-1] < 0.5 * losses[0]
    probs = Sampler(seed=9).sample_unique(120)
    Phi = feature_matrix(p.tree for p in probs)
    rng = np.random.default_rng(1)
    pairs = random_pairs(120, 400, rng)
    y = simulate_judgments(0.15 * Phi[:, 3] - 1.5, pairs, rng)
    m = TreeBT([p.tree for p in probs], encoder=enc).fit_lp_ft(pairs, y)
    assert m.freeze_encoder is False
    from scipy.stats import spearmanr
    assert spearmanr(m.score(), Phi[:, 3])[0] > 0.7
