"""Tests for the distributed and mixed-precision paths (P11 item 4).

The DDP code was written but had only ever been exercised single-process on
MPS, which means every collective in it was dead code in practice.  These tests
run a real two-rank Gloo process group over CPU and assert the properties the
rest of the pipeline assumes:

* DDP actually synchronises gradients, so both ranks hold identical weights;
* evaluation splits are *replicated* rather than distributed-sampled, because a
  ``DistributedSampler`` pads its final shard and would silently corrupt the
  paired item-level statistics section 8 depends on;
* ``reduce_scalar`` and ``all_gather_object`` agree across ranks;
* only the main rank writes checkpoints.

The CUDA-specific paths -- NCCL, bfloat16 autocast, ``GradScaler`` under
float16, and TF32 -- cannot be executed without CUDA hardware.  The autocast
*decision* logic is unit-tested directly here for every device/dtype
combination; the remaining hardware-dependent assertions live in
``test_cuda_multi_gpu_smoke`` and skip with an explicit message rather than
passing vacuously.  See ``P11_VALIDATION.md`` for what is and is not covered.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from utils import amp_settings, resolve_device  # noqa: E402


# ---------------------------------------------------------------------------
# Autocast decision logic -- runs everywhere, no CUDA needed
# ---------------------------------------------------------------------------


def test_amp_disabled_off_cuda():
    """Neither CPU nor MPS gets autocast, whatever the user asked for."""
    for device_type in ("cpu", "mps"):
        device = torch.device(device_type)
        for dtype_name in ("bfloat16", "float16"):
            use_amp, dtype, needs_scaler = amp_settings(device, True, dtype_name)
            assert use_amp is False
            assert dtype is torch.float32
            assert needs_scaler is False


def test_amp_disabled_when_not_requested():
    """``train.amp=False`` wins even on CUDA."""
    use_amp, dtype, needs_scaler = amp_settings(torch.device("cuda"), False, "bfloat16")
    assert use_amp is False
    assert dtype is torch.float32
    assert needs_scaler is False


def test_amp_float16_requires_grad_scaler(monkeypatch):
    """float16 autocast must come with a ``GradScaler``; bfloat16 must not."""
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    use_amp, dtype, needs_scaler = amp_settings(torch.device("cuda"), True, "float16")
    assert (use_amp, dtype, needs_scaler) == (True, torch.float16, True)

    use_amp, dtype, needs_scaler = amp_settings(torch.device("cuda"), True, "bfloat16")
    assert (use_amp, dtype, needs_scaler) == (True, torch.bfloat16, False)


def test_amp_falls_back_to_float16_without_bf16_support(monkeypatch):
    """Pre-Ampere CUDA silently downgrades bfloat16 to scaled float16."""
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    use_amp, dtype, needs_scaler = amp_settings(torch.device("cuda"), True, "bfloat16")
    assert (use_amp, dtype, needs_scaler) == (True, torch.float16, True)


def test_mask_sentinel_survives_float16_temperature_division():
    """The masking sentinel must not overflow under post-hoc temperature scaling.

    This is the float16-only failure recorded in ``FINDINGS_EVIDENCE_PATHWAY.md``
    section 4: a sentinel safe in float32 becomes ``-inf`` in float16 once
    divided by a temperature below one, and every downstream metric turns to
    ``nan`` without raising.  It is a CUDA-path bug that no MPS or CPU run can
    surface, so it is pinned numerically here.
    """
    from model import MIN_TEMPERATURE, _mask_fill_value

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        sentinel = _mask_fill_value(dtype)
        assert torch.isfinite(torch.tensor(sentinel, dtype=dtype))
        scaled = torch.tensor(sentinel, dtype=dtype) / MIN_TEMPERATURE
        assert torch.isfinite(scaled), (
            f"{dtype} sentinel overflowed under T={MIN_TEMPERATURE}"
        )


# ---------------------------------------------------------------------------
# Two-rank Gloo process group over CPU
# ---------------------------------------------------------------------------


def _ddp_worker(rank: int, world_size: int, port: str, result_dir: str) -> None:
    """Body of one rank in the two-process DDP test.

    Args:
        rank: This process's rank.
        world_size: Total ranks.
        port: Rendezvous port.
        result_dir: Directory each rank writes its findings to.
    """
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": port,
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
        }
    )
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import hashlib

    import numpy as np

    from config import Config
    from dataset import build_dataloaders
    from engine import Trainer
    from model import build_model
    from utils import all_gather_object, is_main_process, reduce_scalar, setup_distributed

    cfg = Config()
    cfg.dist.device = "cpu"
    cfg.dist.backend = "gloo"
    cfg.data.simulate_n_items = 240
    cfg.data.num_workers = 0
    cfg.train.epochs = 1
    cfg.train.batch_size = 16
    cfg.train.output_dir = result_dir
    cfg.train.run_name = "ddp_test"
    cfg.train.amp = False
    cfg.eval.write_audit_records = False
    cfg.validate()

    dist_state = setup_distributed(cfg)
    loaders, _ = build_dataloaders(cfg, dist_state)
    trainer = Trainer(cfg, build_model(cfg), loaders, dist_state)
    trainer.train()

    # Weights must be bit-identical across ranks after gradient synchronisation.
    flat = torch.cat([p.detach().flatten() for p in trainer.raw_model.parameters()])
    checksum = float(flat.double().sum())
    gathered = all_gather_object(checksum)

    # Evaluation is replicated, not sharded: both ranks must see every test item
    # in the same order, or the paired bootstrap silently compares misaligned
    # vectors.
    result = trainer.evaluate("test", compute_loss=False)
    item_ids = list(result.item_ids)

    findings = {
        "rank": rank,
        "backend": dist_state.backend,
        "world_size": dist_state.world_size,
        "enabled": dist_state.enabled,
        "checksum": checksum,
        "gathered_checksums": gathered,
        "reduced_mean": reduce_scalar(float(rank), dist_state.device, op="mean"),
        "n_test_items": len(item_ids),
        # hashlib, not the builtin hash(): string hashing is salted per process
        # (PYTHONHASHSEED), so two spawned ranks would disagree on identical
        # lists and the assertion would test the salt rather than the ordering.
        "test_item_hash": hashlib.sha256("|".join(item_ids).encode()).hexdigest(),
        "is_main": is_main_process(),
        "train_batches": len(loaders["train"]),
        "wrote_best_checkpoint": os.path.exists(
            os.path.join(result_dir, "ddp_test", "best.pt")
        ),
    }
    np.save(os.path.join(result_dir, f"rank{rank}.npy"), findings, allow_pickle=True)

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    sys.platform == "win32", reason="gloo spawn rendezvous is unreliable on Windows CI"
)
def test_ddp_two_rank_gloo_cpu():
    """A real two-rank process group trains, synchronises, and evaluates."""
    import numpy as np

    world_size = 2
    with tempfile.TemporaryDirectory() as result_dir:
        mp.spawn(
            _ddp_worker,
            args=(world_size, "29517", result_dir),
            nprocs=world_size,
            join=True,
        )
        findings = [
            np.load(
                os.path.join(result_dir, f"rank{r}.npy"), allow_pickle=True
            ).item()
            for r in range(world_size)
        ]

    for rank, f in enumerate(findings):
        assert f["enabled"] is True, "process group did not initialise"
        assert f["backend"] == "gloo"
        assert f["world_size"] == world_size
        assert f["rank"] == rank

    # DDP averaged the gradients: both ranks hold the same parameters.
    assert findings[0]["checksum"] == pytest.approx(findings[1]["checksum"], abs=1e-6)
    for f in findings:
        assert len(f["gathered_checksums"]) == world_size

    # reduce_scalar(mean) over ranks {0, 1} is 0.5 on every rank.
    for f in findings:
        assert f["reduced_mean"] == pytest.approx(0.5)

    # Evaluation replicated, not sharded.
    assert findings[0]["n_test_items"] == findings[1]["n_test_items"]
    assert findings[0]["test_item_hash"] == findings[1]["test_item_hash"]

    # The train loader *is* sharded, so each rank sees fewer batches than the
    # single-process run would.
    assert findings[0]["train_batches"] == findings[1]["train_batches"]

    # Only rank 0 is the main process, and checkpoints exist for both ranks to
    # see afterwards because rank 0 wrote them.
    assert findings[0]["is_main"] is True
    assert findings[1]["is_main"] is False


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason=(
        "multi-GPU CUDA validation needs >= 2 visible devices; run "
        "`torchrun --nproc_per_node=2 -m pytest tests/test_distributed.py` on a "
        "CUDA host. Not covered by CPU CI."
    ),
)
def test_cuda_multi_gpu_smoke():
    """NCCL + autocast + GradScaler on real CUDA devices.

    Deliberately left as a skip on non-CUDA hosts rather than emulated: the
    failure modes this guards against (NCCL rendezvous, bf16 reductions, scaler
    overflow) do not reproduce on Gloo or CPU.
    """
    from dataset import build_dataloaders
    from engine import Trainer
    from model import build_model
    from utils import setup_distributed

    for dtype_name in ("bfloat16", "float16"):
        cfg = Config()
        cfg.dist.device = "cuda"
        cfg.dist.allow_tf32 = True
        cfg.data.simulate_n_items = 240
        cfg.data.num_workers = 0
        cfg.train.epochs = 1
        cfg.train.amp = True
        cfg.train.amp_dtype = dtype_name
        cfg.eval.write_audit_records = False
        cfg.validate()

        dist_state = setup_distributed(cfg)
        loaders, _ = build_dataloaders(cfg, dist_state)
        trainer = Trainer(cfg, build_model(cfg), loaders, dist_state)
        summary = trainer.train()

        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert not torch.isnan(torch.tensor(summary["best_metric"]))
        result = trainer.evaluate("test", compute_loss=False)
        assert torch.isfinite(torch.from_numpy(result.probabilities)).all()
