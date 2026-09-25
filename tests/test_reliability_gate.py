import numpy as np
import unittest
from scipy.optimize import check_grad
from phgeofuse.reliability_gate import ReliabilityGate


def test_objective_gradient_with_missing_retrieval():
    rng = np.random.default_rng(2)
    x = np.column_stack([np.ones(19), rng.normal(size=(19, 3))])
    experts = rng.normal(size=(19, 3))
    available = np.ones((19, 3), bool)
    available[:7, 1:] = False
    y = rng.normal(size=19)
    w = rng.normal(size=12)
    args = (x, experts, available, y, .03)
    error = check_grad(lambda z: ReliabilityGate.objective(z, *args)[0],
                       lambda z: ReliabilityGate.objective(z, *args)[1], w)
    assert error < 1e-6


def test_learns_when_to_trust_expert_and_ignores_unavailable_values():
    rng = np.random.default_rng(42)
    z = rng.normal(size=(300, 1))
    experts = np.column_stack([np.full(300, 4.), np.full(300, 9.)])
    y = 4 + 5 / (1 + np.exp(-3 * z[:, 0]))
    available = np.ones_like(experts, bool)
    model = ReliabilityGate(.001).fit(z[:200], experts[:200], available[:200], y[:200])
    prediction = model.predict(z[200:], experts[200:], available[200:])
    assert np.sqrt(np.mean((prediction - y[200:])**2)) < .1
    available[200:, 1] = False
    experts[200:, 1] = 1e9
    assert np.allclose(model.predict(z[200:], experts[200:], available[200:]), 4.)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([
        unittest.FunctionTestCase(test_objective_gradient_with_missing_retrieval),
        unittest.FunctionTestCase(test_learns_when_to_trust_expert_and_ignores_unavailable_values),
    ])
