"""Isolate gradient-based support adaptation from meta-training state."""
from contextlib import contextmanager

import torch


@contextmanager
def evaluation_state(model, support_generator, *, restore_parameters=False):
    modes = [(module, module.training) for module in model.modules()]
    parameters = [p for p in model.parameters() if p.requires_grad]
    values = [p.detach().clone() for p in parameters] if restore_parameters else []
    gradients = [p.grad for p in parameters]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    support_state = support_generator.get_state()
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})

    def restore():
        with torch.no_grad():
            for parameter, value in zip(parameters, values):
                parameter.copy_(value)
            for buffer, value in buffers:
                buffer.copy_(value)

    # DataLoader iteration consumes the global RNG even when shuffle=False.
    # Keep gradients enabled because support-set adaptation still needs them.
    with torch.random.fork_rng(devices=devices):
        try:
            model.eval()
            yield restore
        finally:
            restore()
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient
            support_generator.set_state(support_state)
            for module, mode in modes:
                module.training = mode
