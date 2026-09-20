"""SPD's Muon/AdamW and EMA, adapted to ABC's single-optimizer loop."""

from contextlib import contextmanager
import math

import torch


class MuonAdamW(torch.optim.Optimizer):
    """One optimizer/checkpoint interface over disjoint matrix and vector groups."""

    def __init__(self, model, config):
        matrices, other = [], []
        for parameter in model.parameters():
            if parameter.requires_grad:
                (matrices if parameter.ndim == 2 else other).append(parameter)
        if not matrices or not other:
            raise ValueError("SPD requires both matrix and non-matrix trainable parameters")
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("SPD requires PyTorch with torch.optim.Muon")
        self.muon = torch.optim.Muon(
            matrices, lr=config.learning_rate, momentum=config.muon_momentum,
            weight_decay=config.weight_decay,
        )
        self.adamw = torch.optim.AdamW(
            other, lr=config.learning_rate, weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon,
        )
        super().__init__(self.muon.param_groups + self.adamw.param_groups, {})

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self.adamw.step()
        return loss

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict):
        if set(state_dict) != {"muon", "adamw"}:
            raise ValueError("SPD resume requires both Muon and AdamW optimizer states")
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        # The child optimizers replace their group dictionaries during loading.
        self.param_groups = self.muon.param_groups + self.adamw.param_groups


class EMA:
    def __init__(self, model, half_life_steps):
        if not math.isfinite(half_life_steps) or half_life_steps <= 0:
            raise ValueError("EMA half-life must be finite and positive")
        self.decay = math.exp(math.log(0.5) / half_life_steps)
        self.model = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        for name, parameter in model.named_parameters():
            if name in self.model:
                self.model[name].lerp_(parameter.detach(), 1 - self.decay)

    def state_dict(self):
        return {"decay": self.decay, "model": self.model}

    def load_state_dict(self, state):
        if state.get("decay") != self.decay or set(state.get("model", {})) != set(self.model):
            raise ValueError("SPD EMA configuration or parameter set changed on resume")
        for name, target in self.model.items():
            source = state["model"][name]
            if source.shape != target.shape or not torch.isfinite(source).all():
                raise ValueError(f"invalid SPD EMA tensor: {name}")
            target.copy_(source.to(device=target.device, dtype=target.dtype))

    @contextmanager
    def apply(self, model):
        parameters = dict(model.named_parameters())
        saved = {name: parameters[name].detach().clone() for name in self.model}
        try:
            with torch.no_grad():
                for name, value in self.model.items():
                    parameters[name].copy_(value)
            yield
        finally:
            with torch.no_grad():
                for name, value in saved.items():
                    parameters[name].copy_(value)
