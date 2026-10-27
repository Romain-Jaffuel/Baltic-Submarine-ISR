"""Deterministic, independent streams."""

import numpy as np

from baltic_isr import seeds


def test_key_is_stable_across_processes() -> None:
    """Not Python's hash, which is salted per interpreter."""
    assert seeds.key("traffic") == seeds.key("traffic")
    assert seeds.key("traffic") == 3815526598893009650


def test_different_names_give_different_streams() -> None:
    a = seeds.generator(7, "traffic").random(16)
    b = seeds.generator(7, "detector").random(16)
    assert not np.allclose(a, b)


def test_same_name_and_root_repeats() -> None:
    a = seeds.generator(7, "traffic").random(16)
    b = seeds.generator(7, "traffic").random(16)
    assert np.allclose(a, b)


def test_different_roots_give_different_streams() -> None:
    a = seeds.generator(7, "traffic").random(16)
    b = seeds.generator(8, "traffic").random(16)
    assert not np.allclose(a, b)


def test_torch_generator_is_reproducible() -> None:
    import torch

    a = torch.rand(8, generator=seeds.torch_generator(3, "policy"))
    b = torch.rand(8, generator=seeds.torch_generator(3, "policy"))
    assert torch.allclose(a, b)


def test_adding_a_subsystem_does_not_shift_the_others() -> None:
    """Streams are independent, so a new subsystem costs nothing in churn."""
    before = seeds.generator(1, "traffic").random(8)
    seeds.generator(1, "a_brand_new_subsystem").random(8)
    after = seeds.generator(1, "traffic").random(8)
    assert np.allclose(before, after)
