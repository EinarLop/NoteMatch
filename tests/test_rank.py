import numpy as np
import pytest

from notematch.recommend import rank

# Worked example from design.md 5.3: topics = [backpropagation, convolution layers]
S = np.array([
    [0.62, 0.84],   # 0 convolutional_networks: covers both
    [0.81, 0.41],   # 1 neural_networks: covers one
    [0.30, 0.20],   # 2 reinforcement_learning: covers none
])


def test_worked_example_order_and_scores():
    order, scores, covered = rank(S, tau_topic=0.55, tau_min=0.30, max_results=3)
    assert order == [0, 1]                                   # reinforcement_learning excluded
    assert scores[:2] == pytest.approx([0.73, 0.405])
    assert covered.tolist() == [[True, True], [True, False], [False, False]]


def test_max_results_caps_output():
    order, _, _ = rank(S, tau_topic=0.55, tau_min=0.30, max_results=1)
    assert order == [0]


def test_coverage_beats_single_strong_match():
    # One very strong topic loses to covering both topics decently.
    S2 = np.array([[0.95, 0.10], [0.60, 0.60]])
    order, _, _ = rank(S2, tau_topic=0.55, tau_min=0.0, max_results=3)
    assert order == [1, 0]
