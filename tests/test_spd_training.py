from copy import deepcopy

import torch

from abc_minimal.checkpointing import load_checkpoint, restore_training_state, save_checkpoint
from abc_minimal.config import OptimConfig
from abc_minimal.spd_optim import EMA, MuonAdamW


def test_muon_adamw_checkpoint_resume_preserves_next_update(tmp_path):
    torch.manual_seed(19)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 2))
    config = OptimConfig(learning_rate=0.01, lr_warmup_steps=4, weight_decay=0.1)
    optimizer = MuonAdamW(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / 4, 1))
    ema = EMA(model, 20)
    inputs, target = torch.randn(3, 4), torch.randn(3, 2)

    def update(net, opt, sched, moving):
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(net(inputs), target).backward()
        opt.step()
        sched.step()
        moving.update(net)

    update(model, optimizer, scheduler, ema)
    path = tmp_path / "last.pt"
    save_checkpoint(path, module=model, optimizer=optimizer, scheduler=scheduler,
                    global_step=1, norm_stats={}, extra_state={"ema": ema.state_dict()})
    checkpoint, step = load_checkpoint(path)
    restored = deepcopy(model)
    restored.load_state_dict(checkpoint["model"])
    resumed_optimizer = MuonAdamW(restored, config)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda step: min((step + 1) / 4, 1))
    restore_training_state(checkpoint, optimizer=resumed_optimizer, scheduler=resumed_scheduler,
                           resume_step=step, rank=0)
    resumed_ema = EMA(restored, 20)
    resumed_ema.load_state_dict(checkpoint["ema"])
    update(model, optimizer, scheduler, ema)
    update(restored, resumed_optimizer, resumed_scheduler, resumed_ema)
    torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)
    with ema.apply(model), resumed_ema.apply(restored):
        torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)
    torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)
