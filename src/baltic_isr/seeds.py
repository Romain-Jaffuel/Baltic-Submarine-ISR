"""Deterministic seeds, one independent stream per named subsystem.

A run is reproducible bit for bit from its root seed and its configuration
fingerprint. That only holds if every stream is derived from the root by a
function that does not change between processes -- so names are hashed with
BLAKE2b, never with ``hash``, which is salted per interpreter.

Streams are independent: adding a subsystem does not shift the numbers any
other subsystem draws.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

_DIGEST_BYTES = 8


def key(name: str) -> int:
    """Return a stable 64-bit integer for a subsystem name.

    Parameters
    ----------
    name : str
        Subsystem name, for example ``"traffic"`` or ``"detector"``.

    Returns
    -------
    int
        A value that is identical in every process and every platform.
    """
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=_DIGEST_BYTES).digest()
    return int.from_bytes(digest, "big")


def sequence(root: int, name: str) -> np.random.SeedSequence:
    """Return the seed sequence of one named subsystem.

    Parameters
    ----------
    root : int
        Root seed, from configuration.
    name : str
        Subsystem name.

    Returns
    -------
    numpy.random.SeedSequence
        An independent sequence.
    """
    return np.random.SeedSequence([root, key(name)])


def generator(root: int, name: str) -> np.random.Generator:
    """Return a NumPy generator for one named subsystem.

    Parameters
    ----------
    root : int
        Root seed.
    name : str
        Subsystem name.

    Returns
    -------
    numpy.random.Generator
        A fresh generator.
    """
    return np.random.default_rng(sequence(root, name))


def torch_generator(root: int, name: str, device: torch.device | str = "cpu") -> torch.Generator:
    """Return a torch generator for one named subsystem.

    The training path draws every random number from one of these, so that a
    run on the GPU and the same run on the CPU differ only by floating-point
    association, never by which numbers were drawn.

    Parameters
    ----------
    root : int
        Root seed.
    name : str
        Subsystem name.
    device : torch.device or str
        Device the generator draws for.

    Returns
    -------
    torch.Generator
        A seeded generator.
    """
    state = sequence(root, name).generate_state(1, dtype=np.uint32)
    handle = torch.Generator(device=device)
    handle.manual_seed(int(state[0]))
    return handle
