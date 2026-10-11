"""Transactional in-place state for opt-in probes; never promotes a parent."""
import copy
import hashlib
import random
from contextlib import contextmanager

import numpy as np
import torch

from .feature_contract import ContractError


def _feed(h, value):
    if isinstance(value, torch.Tensor):
        x = value.detach().cpu().contiguous()
        h.update(str((x.dtype, tuple(x.shape))).encode())
        h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        h.update(str((value.dtype, value.shape)).encode())
        h.update(value.tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            _feed(h, key)
            _feed(h, value[key])
    elif isinstance(value, (tuple, list)):
        h.update(str(type(value)).encode())
        for item in value:
            _feed(h, item)
    else:
        h.update(repr(value).encode())
    h.update(b"\0")


def digest_state(value):
    h = hashlib.sha256()
    _feed(h, value)
    return h.hexdigest()


def model_digest(model):
    return digest_state({**dict(model.named_parameters()), **dict(model.named_buffers())})


def rng_state(generators=()):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            "generators": [g.get_state().clone() for g in generators]}


def restore_rng(state, generators=()):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    for generator, value in zip(generators, state["generators"]):
        generator.set_state(value)


@contextmanager
def seeded_stream(seed):
    before = rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        restore_rng(before)


class TrainingSnapshot:
    """CPU copies of parameters/buffers, including nonpersistent buffers.

    Stateful optimizer/scheduler/scaler/sampler objects must expose state_dict /
    load_state_dict. Explicit torch generators are included. No iterable/sampler
    cursor is guessed. The caller must not add/remove parameters or modules.
    Quantized tensors are copied in-place, not moved/reloaded through .to().
    """
    def __init__(self, model, *, stateful=(), generators=()):
        self.model, self.stateful, self.generators = model, tuple(stateful), tuple(generators)
        for obj in self.stateful:
            if not callable(getattr(obj, "state_dict", None)) or not callable(getattr(obj, "load_state_dict", None)):
                raise ContractError("All external training/sampler state must be explicitly restorable")
        self.parameters = dict(model.named_parameters())
        self.buffers = dict(model.named_buffers())
        self.values = {k: v.detach().cpu().clone() for k, v in {**self.parameters, **self.buffers}.items()}
        self.grads = {k: None if v.grad is None else v.grad.detach().cpu().clone()
                      for k, v in self.parameters.items()}
        self.requires = {k: v.requires_grad for k, v in self.parameters.items()}
        self.modes = {k: v.training for k, v in model.named_modules()}
        self.states = copy.deepcopy([obj.state_dict() for obj in self.stateful])
        self.rng = rng_state(self.generators)
        self.before = self.fingerprint()

    def fingerprint(self):
        return digest_state({"values": {**dict(self.model.named_parameters()), **dict(self.model.named_buffers())},
            "grads": {k: v.grad for k, v in self.model.named_parameters()},
            "requires": {k: v.requires_grad for k, v in self.model.named_parameters()},
            "modes": {k: v.training for k, v in self.model.named_modules()},
            "states": [obj.state_dict() for obj in self.stateful], "rng": rng_state(self.generators)})

    def restore(self):
        if (dict(self.model.named_parameters()).keys() != self.parameters.keys()
                or dict(self.model.named_buffers()).keys() != self.buffers.keys()
                or dict(self.model.named_modules()).keys() != self.modes.keys()):
            raise ContractError("Runtime structure changed during probe; stop before another sibling")
        if any(p is not self.parameters[name] for name, p in self.model.named_parameters()):
            raise ContractError("Parameter identity changed during probe; cannot restore an attached optimizer safely")
        # Modules can replace a registered buffer (e.g. a cache) rather than
        # mutate it. Restore the original registered object as well as its bytes.
        for name, value in self.buffers.items():
            parent, _, local = name.rpartition(".")
            module = self.model.get_submodule(parent) if parent else self.model
            module._buffers[local] = value
        with torch.no_grad():
            for name, value in {**self.parameters, **self.buffers}.items():
                value.copy_(self.values[name])
        for name, parameter in self.parameters.items():
            parameter.requires_grad_(self.requires[name])
            gradient = self.grads[name]
            parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()
        for name, module in self.model.named_modules():
            module.training = self.modes[name]
        for obj, state in zip(self.stateful, self.states):
            obj.load_state_dict(copy.deepcopy(state))
        restore_rng(self.rng, self.generators)
        if self.fingerprint() != self.before:
            raise ContractError("Full training-state restoration failed; stop before another sibling")


@contextmanager
def restored_training_state(model, *, stateful=(), generators=()):
    snapshot = TrainingSnapshot(model, stateful=stateful, generators=generators)
    try:
        yield snapshot
    finally:
        snapshot.restore()
