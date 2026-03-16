"""
Unit tests for Phase 1 training utilities:
  - build_optimizer (training/scheduler.py)
  - build_scheduler (training/scheduler.py)
  - EarlyStopping (training/scheduler.py)
  - Trainer (training/trainer.py)

All tests use synthetic tensors and a tiny dummy model — no nuScenes data required.
Safe to run in CI without GPU.
"""

from __future__ import annotations

import os
import tempfile
from typing import Iterator
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_model(num_classes: int = 3) -> nn.Module:
    """A minimal two-layer model with weight-decay-eligible and non-eligible params."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(16, 32),   # weight: 2D (decay), bias: 1D (no decay)
        nn.ReLU(),
        nn.Linear(32, num_classes),  # weight: 2D (decay), bias: 1D (no decay)
    )


def _make_loader(num_samples: int = 8, num_classes: int = 3, batch_size: int = 4):
    """
    Build a DataLoader whose collate_fn returns the (images, targets) format
    that Trainer expects: targets is a list of dicts with key 'labels'.
    Each image gets exactly one label to avoid the label-count vs batch-size
    mismatch described in the code-review notes (bug #9).
    """
    images = torch.randn(num_samples, 16)
    label_vals = torch.randint(0, num_classes, (num_samples,))

    class _SingleLabelDataset(torch.utils.data.Dataset):
        def __len__(self):
            return num_samples

        def __getitem__(self, idx):
            return images[idx], {'labels': label_vals[idx:idx + 1]}

    def _collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        targets = [b[1] for b in batch]
        return imgs, targets

    return DataLoader(
        _SingleLabelDataset(),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
    )


# ── build_optimizer ────────────────────────────────────────────────────────────

class TestBuildOptimizer:

    def test_returns_adamw_by_default(self):
        from training.scheduler import build_optimizer
        model = _tiny_model()
        opt = build_optimizer(model)
        assert isinstance(opt, torch.optim.AdamW)

    def test_returns_sgd_when_requested(self):
        from training.scheduler import build_optimizer
        model = _tiny_model()
        opt = build_optimizer(model, optimizer_type="sgd", lr=0.01)
        assert isinstance(opt, torch.optim.SGD)

    def test_two_param_groups_always(self):
        """build_optimizer must always produce exactly 2 param groups."""
        from training.scheduler import build_optimizer
        model = _tiny_model()
        opt = build_optimizer(model)
        assert len(opt.param_groups) == 2

    def test_decay_group_has_nonzero_weight_decay(self):
        from training.scheduler import build_optimizer
        model = _tiny_model()
        opt = build_optimizer(model, weight_decay=1e-4)
        # First group: params with ndim >= 2 (the weight matrices)
        decay_group = opt.param_groups[0]
        assert decay_group['weight_decay'] == pytest.approx(1e-4)

    def test_no_decay_group_has_zero_weight_decay(self):
        from training.scheduler import build_optimizer
        model = _tiny_model()
        opt = build_optimizer(model, weight_decay=1e-4)
        no_decay_group = opt.param_groups[1]
        assert no_decay_group['weight_decay'] == pytest.approx(0.0)

    def test_decay_group_contains_weight_matrices_only(self):
        """2D parameters (weight matrices) must land in the decay group."""
        from training.scheduler import build_optimizer
        model = _tiny_model()
        opt = build_optimizer(model)
        decay_params = opt.param_groups[0]['params']
        for p in decay_params:
            assert p.ndim >= 2, f"A 1D param ended up in the decay group: {p.shape}"

    def test_no_decay_group_contains_biases_only(self):
        """1D parameters (biases) must land in the no-decay group."""
        from training.scheduler import build_optimizer
        model = _tiny_model()
        opt = build_optimizer(model)
        no_decay_params = opt.param_groups[1]['params']
        for p in no_decay_params:
            assert p.ndim < 2, f"A >=2D param ended up in the no-decay group: {p.shape}"

    def test_lr_is_respected(self):
        from training.scheduler import build_optimizer
        model = _tiny_model()
        opt = build_optimizer(model, lr=5e-4)
        for group in opt.param_groups:
            assert group['lr'] == pytest.approx(5e-4)

    def test_all_parameters_covered(self):
        """Every model parameter must appear in exactly one param group."""
        from training.scheduler import build_optimizer
        model = _tiny_model()
        opt = build_optimizer(model)
        all_in_groups = [p for g in opt.param_groups for p in g['params']]
        model_params = list(model.parameters())
        assert len(all_in_groups) == len(model_params), \
            "Parameter count mismatch between model and optimizer groups"

    def test_sgd_momentum_is_set(self):
        from training.scheduler import build_optimizer
        model = _tiny_model()
        opt = build_optimizer(model, optimizer_type="sgd", lr=0.01, momentum=0.9)
        for group in opt.param_groups:
            assert group['momentum'] == pytest.approx(0.9)

    def test_optimizer_step_does_not_crash(self):
        """A full forward+backward+step cycle must complete without error."""
        from training.scheduler import build_optimizer
        torch.manual_seed(42)
        model = _tiny_model()
        opt = build_optimizer(model)
        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()
        opt.step()  # must not raise


# ── build_scheduler ────────────────────────────────────────────────────────────

class TestBuildScheduler:

    def _get_optimizer(self):
        model = _tiny_model()
        return torch.optim.AdamW(model.parameters(), lr=1e-3)

    def test_cosine_returns_cosine_scheduler(self):
        from training.scheduler import build_scheduler
        opt = self._get_optimizer()
        sched = build_scheduler(opt, scheduler_type="cosine", epochs=10)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_plateau_returns_reduce_lr_on_plateau(self):
        from training.scheduler import build_scheduler
        opt = self._get_optimizer()
        sched = build_scheduler(opt, scheduler_type="plateau")
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_cosine_lr_decreases_over_epochs(self):
        """CosineAnnealing LR should be lower at epoch T/2 than at epoch 0."""
        from training.scheduler import build_scheduler
        opt = self._get_optimizer()
        sched = build_scheduler(opt, scheduler_type="cosine", epochs=20)
        lr_start = opt.param_groups[0]['lr']
        for _ in range(10):
            opt.step()
            sched.step()
        lr_mid = opt.param_groups[0]['lr']
        assert lr_mid < lr_start, "CosineAnnealing LR did not decrease over 10 steps"

    def test_cosine_min_lr_not_below_zero(self):
        """CosineAnnealing must not push LR below 0."""
        from training.scheduler import build_scheduler
        opt = self._get_optimizer()
        sched = build_scheduler(opt, scheduler_type="cosine", epochs=5)
        for _ in range(5):
            opt.step()
            sched.step()
        lr_final = opt.param_groups[0]['lr']
        assert lr_final >= 0.0

    def test_plateau_reduces_lr_after_patience(self):
        """ReduceLROnPlateau reduces LR after `patience + 1` stagnant steps.

        PyTorch's ReduceLROnPlateau starts counting from 1 on the first
        non-improving epoch, so it requires patience + 1 total stagnant
        calls before the LR reduction fires. Each call must be preceded
        by optimizer.step() per the PyTorch 1.1+ ordering requirement.
        """
        from training.scheduler import build_scheduler
        opt = self._get_optimizer()
        patience = 2
        sched = build_scheduler(opt, scheduler_type="plateau", patience=patience)
        initial_lr = opt.param_groups[0]['lr']
        stagnant_loss = 1.0
        # patience + 2 calls: first is a "best" epoch, then patience + 1
        # non-improving epochs trigger the reduction.
        for _ in range(patience + 2):
            opt.step()
            sched.step(stagnant_loss)
        reduced_lr = opt.param_groups[0]['lr']
        assert reduced_lr < initial_lr, \
            "ReduceLROnPlateau did not reduce LR after patience exceeded"

    def test_cosine_scheduler_epochs_param(self):
        """T_max of CosineAnnealingLR must equal the `epochs` argument."""
        from training.scheduler import build_scheduler
        opt = self._get_optimizer()
        sched = build_scheduler(opt, scheduler_type="cosine", epochs=50)
        assert sched.T_max == 50


# ── EarlyStopping ──────────────────────────────────────────────────────────────

class TestEarlyStopping:

    @pytest.fixture
    def tmpdir_path(self, tmp_path):
        return str(tmp_path / "best_model.pt")

    def test_should_stop_false_initially(self, tmpdir_path):
        from training.scheduler import EarlyStopping
        es = EarlyStopping(patience=3, ckpt_path=tmpdir_path)
        assert es.should_stop is False

    def test_counter_starts_at_zero(self, tmpdir_path):
        from training.scheduler import EarlyStopping
        es = EarlyStopping(patience=3, ckpt_path=tmpdir_path)
        assert es.counter == 0

    def test_improvement_resets_counter(self, tmpdir_path):
        from training.scheduler import EarlyStopping
        model = _tiny_model()
        es = EarlyStopping(patience=3, ckpt_path=tmpdir_path)
        es(1.0, model)  # improvement
        es(0.9, model)  # improvement
        assert es.counter == 0

    def test_no_improvement_increments_counter(self, tmpdir_path):
        from training.scheduler import EarlyStopping
        model = _tiny_model()
        es = EarlyStopping(patience=3, ckpt_path=tmpdir_path)
        es(1.0, model)  # first call — improvement from inf
        es(1.0, model)  # no improvement
        es(1.0, model)  # no improvement
        assert es.counter == 2

    def test_triggers_after_patience_steps(self, tmpdir_path):
        from training.scheduler import EarlyStopping
        model = _tiny_model()
        es = EarlyStopping(patience=3, ckpt_path=tmpdir_path)
        es(1.0, model)  # improvement
        es(1.0, model)  # +1
        es(1.0, model)  # +2
        es(1.0, model)  # +3 → should_stop
        assert es.should_stop is True

    def test_does_not_trigger_before_patience_exhausted(self, tmpdir_path):
        from training.scheduler import EarlyStopping
        model = _tiny_model()
        es = EarlyStopping(patience=5, ckpt_path=tmpdir_path)
        es(1.0, model)
        for _ in range(4):  # 4 non-improvements, patience=5 → not yet
            es(1.0, model)
        assert es.should_stop is False

    def test_checkpoint_saved_on_improvement(self, tmpdir_path):
        from training.scheduler import EarlyStopping
        model = _tiny_model()
        es = EarlyStopping(patience=3, ckpt_path=tmpdir_path)
        es(1.0, model)
        assert os.path.exists(tmpdir_path), "Checkpoint file not written on improvement"

    def test_best_loss_updated_on_improvement(self, tmpdir_path):
        from training.scheduler import EarlyStopping
        model = _tiny_model()
        es = EarlyStopping(patience=3, min_delta=1e-4, ckpt_path=tmpdir_path)
        es(0.5, model)
        assert es.best_loss == pytest.approx(0.5)

    def test_min_delta_prevents_trivial_improvement(self, tmpdir_path):
        """An improvement smaller than min_delta must not reset the counter."""
        from training.scheduler import EarlyStopping
        model = _tiny_model()
        min_delta = 0.1
        es = EarlyStopping(patience=3, min_delta=min_delta, ckpt_path=tmpdir_path)
        es(1.0, model)   # improvement from inf → counter=0
        es(0.999, model) # delta=0.001 < 0.1 → not an improvement → counter=1
        assert es.counter == 1

    def test_load_best_restores_weights(self, tmpdir_path):
        """load_best must restore the weights saved at the best checkpoint."""
        from training.scheduler import EarlyStopping
        torch.manual_seed(42)
        model = _tiny_model()
        es = EarlyStopping(patience=3, ckpt_path=tmpdir_path)
        # Save a checkpoint of the current weights
        es(1.0, model)
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        # Now change the model's weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)
        # Restore
        es.load_best(model)
        restored = model.state_dict()
        for key in best_weights:
            assert torch.allclose(restored[key], best_weights[key]), \
                f"Restored weight {key} does not match saved checkpoint"

    def test_counter_resets_after_late_improvement(self, tmpdir_path):
        """Counter must reset to 0 when improvement arrives after several stagnant steps."""
        from training.scheduler import EarlyStopping
        model = _tiny_model()
        es = EarlyStopping(patience=5, ckpt_path=tmpdir_path)
        es(1.0, model)   # improvement
        es(1.0, model)   # +1
        es(1.0, model)   # +2
        es(0.5, model)   # improvement → counter resets
        assert es.counter == 0
        assert es.should_stop is False


# ── Trainer ────────────────────────────────────────────────────────────────────

class TestAverageMeter:

    def test_initial_state(self):
        from training.trainer import AverageMeter
        m = AverageMeter()
        assert m.val == 0.0
        assert m.avg == 0.0
        assert m.sum == 0.0
        assert m.count == 0.0

    def test_single_update(self):
        from training.trainer import AverageMeter
        m = AverageMeter()
        m.update(2.0, n=4)
        assert m.val == pytest.approx(2.0)
        assert m.sum == pytest.approx(8.0)
        assert m.count == pytest.approx(4.0)
        assert m.avg == pytest.approx(2.0)

    def test_multiple_updates_weighted_average(self):
        """avg must be sum/count, not a simple mean of val calls."""
        from training.trainer import AverageMeter
        m = AverageMeter()
        m.update(10.0, n=1)
        m.update(0.0, n=9)
        # sum=10, count=10 → avg=1.0
        assert m.avg == pytest.approx(1.0)

    def test_reset_clears_state(self):
        from training.trainer import AverageMeter
        m = AverageMeter()
        m.update(5.0, n=2)
        m.reset()
        assert m.avg == 0.0
        assert m.count == 0.0


class TestTrainer:
    """
    Tests for Trainer use a tiny feedforward model and a synthetic loader
    that produces (image_tensor, list_of_target_dicts) with exactly one label
    per image to avoid the batch-size vs label-count mismatch.
    """

    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        model = _tiny_model(num_classes=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        train_loader = _make_loader(num_samples=8, batch_size=4)
        val_loader = _make_loader(num_samples=4, batch_size=4)
        from training.trainer import Trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            device="cpu",
        )
        return trainer

    def test_train_epoch_returns_loss_and_acc_keys(self, setup):
        metrics = setup.train_epoch(epoch=1)
        assert 'loss' in metrics
        assert 'acc' in metrics

    def test_val_epoch_returns_loss_and_acc_keys(self, setup):
        metrics = setup.val_epoch()
        assert 'loss' in metrics
        assert 'acc' in metrics

    def test_train_epoch_loss_is_positive_float(self, setup):
        metrics = setup.train_epoch(epoch=1)
        assert isinstance(metrics['loss'], float)
        assert metrics['loss'] > 0.0

    def test_val_epoch_loss_is_positive_float(self, setup):
        metrics = setup.val_epoch()
        assert isinstance(metrics['loss'], float)
        assert metrics['loss'] > 0.0

    def test_accuracy_is_in_0_1_range(self, setup):
        train_m = setup.train_epoch(epoch=1)
        val_m = setup.val_epoch()
        assert 0.0 <= train_m['acc'] <= 1.0
        assert 0.0 <= val_m['acc'] <= 1.0

    def test_model_in_train_mode_during_train_epoch(self, setup):
        """Trainer must call model.train() before train_epoch loop."""
        # We verify by checking BN running stats update only in train mode.
        # Simpler proxy: after train_epoch, the model should still be in train mode
        # because val_epoch has not been called yet.
        # Actually train_epoch does not guarantee mode after completion;
        # instead we patch model.train to verify it is called.
        original_train = setup.model.train
        call_log = []
        def patched_train(mode=True):
            call_log.append(mode)
            return original_train(mode)
        setup.model.train = patched_train
        setup.train_epoch(epoch=1)
        assert True in call_log, "model.train(True) was never called inside train_epoch"

    def test_model_in_eval_mode_during_val_epoch(self, setup):
        """Trainer must call model.eval() before val_epoch loop."""
        original_eval = setup.model.eval
        call_log = []
        def patched_eval():
            call_log.append(True)
            return original_eval()
        setup.model.eval = patched_eval
        setup.val_epoch()
        assert len(call_log) > 0, "model.eval() was never called inside val_epoch"

    def test_optimizer_zero_grad_called_each_step(self, setup):
        """zero_grad must be called before each backward pass."""
        call_count = [0]
        original_zero = setup.optimizer.zero_grad
        def patched_zero_grad(*args, **kwargs):
            call_count[0] += 1
            return original_zero(*args, **kwargs)
        setup.optimizer.zero_grad = patched_zero_grad
        setup.train_epoch(epoch=1)
        # 8 samples / batch_size 4 = 2 batches → 2 zero_grad calls
        assert call_count[0] == 2

    def test_parameters_change_after_train_epoch(self, setup):
        """Model weights must differ before and after one training epoch."""
        params_before = [p.detach().clone() for p in setup.model.parameters()]
        setup.train_epoch(epoch=1)
        params_after = list(setup.model.parameters())
        any_changed = any(
            not torch.allclose(b, a.detach())
            for b, a in zip(params_before, params_after)
        )
        assert any_changed, "No model parameters changed after train_epoch"

    def test_val_epoch_no_grad(self, setup):
        """val_epoch must run under torch.no_grad() — parameters must have no grad."""
        setup.val_epoch()
        for p in setup.model.parameters():
            assert p.grad is None or torch.all(p.grad == 0), \
                "A gradient was computed during val_epoch — torch.no_grad() missing?"

    def test_fit_runs_correct_number_of_epochs(self, setup):
        """fit(epochs=N) must call train_epoch exactly N times."""
        call_count = [0]
        original = setup.train_epoch
        def patched(epoch):
            call_count[0] += 1
            return original(epoch)
        setup.train_epoch = patched
        setup.fit(epochs=3)
        assert call_count[0] == 3

    def test_fit_with_cosine_scheduler_does_not_crash(self, setup):
        from training.scheduler import build_scheduler
        sched = build_scheduler(setup.optimizer, scheduler_type="cosine", epochs=2)
        setup.fit(epochs=2, scheduler=sched)  # must not raise

    def test_fit_with_plateau_scheduler_does_not_crash(self, setup):
        from training.scheduler import build_scheduler
        sched = build_scheduler(setup.optimizer, scheduler_type="plateau")
        setup.fit(epochs=2, scheduler=sched)  # must not raise

    def test_fit_stops_early_when_patience_exceeded(self, setup, tmp_path):
        """fit() must honour early_stopping.should_stop and break before max epochs."""
        from training.scheduler import EarlyStopping
        ckpt = str(tmp_path / "es_test.pt")
        # patience=1 with a stagnant loss (random model, same data) should stop fast
        es = EarlyStopping(patience=1, min_delta=1e10, ckpt_path=ckpt)
        call_count = [0]
        original = setup.train_epoch
        def patched(epoch):
            call_count[0] += 1
            return original(epoch)
        setup.train_epoch = patched
        setup.fit(epochs=10, early_stopping=es)
        # With patience=1 and min_delta=1e10, no improvement is ever seen
        # after the first epoch that saves the checkpoint.
        # Should stop well before 10 epochs.
        assert call_count[0] < 10, \
            f"Early stopping did not trigger — ran {call_count[0]} epochs out of 10"

    def test_model_moved_to_device(self, setup):
        """Trainer constructor must move model to the specified device."""
        for p in setup.model.parameters():
            assert str(p.device) == "cpu"
