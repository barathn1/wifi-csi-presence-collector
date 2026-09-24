"""Daily-recalibration wrapper via Reptile (first-order meta-learning) -- a deliberately simplified
stand-in for full second-order MAML, chosen because it needs no second-derivative machinery and is
much cheaper per meta-step while capturing the same core idea: instead of forcing one embedding to be
day-invariant outright, meta-train ACROSS days-as-tasks so the model can adapt in a handful of
gradient steps to a brand-new day using a short calibration walk, then use the adapted (not the
original) weights for that day's actual decisions.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


def reptile_meta_train(model: nn.Module, day_tasks: list[tuple], inner_steps: int = 5,
                        inner_lr: float = 1e-2, meta_lr: float = 0.5, meta_epochs: int = 10,
                        loss_fn: callable = None, device: torch.device = torch.device("cpu")) -> nn.Module:
    """`day_tasks`: list of (support_loader, ) tuples, one per day -- each a small DataLoader of that
    day's windows. For each meta-epoch, for each day: clone current weights, take `inner_steps` SGD
    steps on that day's data alone (the "adapt to this day" simulation), then move the META-weights a
    fraction (`meta_lr`) of the way toward the adapted weights (Reptile's update rule, Nichol et al.
    2018) -- repeated across every day so the FINAL weights sit in a region that's cheap to fine-tune
    from for any day, not just one specific day.
    """
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    model.to(device)
    for _ in range(meta_epochs):
        for (loader,) in day_tasks:
            fast_model = copy.deepcopy(model)
            inner_opt = torch.optim.SGD(fast_model.parameters(), lr=inner_lr)
            for step, (amp, phase, label) in enumerate(loader):
                if step >= inner_steps:
                    break
                amp, phase, label = amp.to(device), phase.to(device), label.to(device)
                inner_opt.zero_grad()
                loss = loss_fn(fast_model(amp, phase), label)
                loss.backward()
                inner_opt.step()
            with torch.no_grad():
                for meta_p, fast_p in zip(model.parameters(), fast_model.parameters()):
                    meta_p.add_(meta_lr * (fast_p - meta_p))
    return model


def adapt_to_day(model: nn.Module, calib_loader, inner_steps: int = 10, inner_lr: float = 1e-2,
                  loss_fn: callable = None, device: torch.device = torch.device("cpu")) -> nn.Module:
    """The actual deployment-time step: given the meta-trained `model` and a short calibration walk's
    worth of labeled windows for a NEW day, return a day-adapted COPY (never mutates `model` itself, so
    the same meta-trained starting point can be re-adapted fresh for every new day)."""
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    adapted = copy.deepcopy(model).to(device)
    opt = torch.optim.SGD(adapted.parameters(), lr=inner_lr)
    for step, (amp, phase, label) in enumerate(calib_loader):
        if step >= inner_steps:
            break
        amp, phase, label = amp.to(device), phase.to(device), label.to(device)
        opt.zero_grad()
        loss = loss_fn(adapted(amp, phase), label)
        loss.backward()
        opt.step()
    return adapted
