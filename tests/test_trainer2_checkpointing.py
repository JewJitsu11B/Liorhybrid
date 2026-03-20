import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="trainer2 import is CUDA-only")
def test_maybe_checkpoint_saves_on_manual_quit_even_when_periodic_disabled():
    from training import trainer2

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SimpleNamespace(
            save_every_windows=0,
            run_dir=tmpdir,
            run_name="manual_quit_test",
            coord_dim_n=8,
        )
        model = SimpleNamespace(
            d_model=16,
            n_layers=2,
            n_heads=2,
            n_attention_layers=1,
            input_embedding=None,
            lm_head=None,
            state_dict=lambda: {"w": torch.ones(1)},
        )
        field = SimpleNamespace()
        memory = SimpleNamespace()
        rotor_state = SimpleNamespace()

        trainer2.maybe_checkpoint(
            window_idx=3,
            epoch_idx=1,
            cfg=cfg,
            model=model,
            field=field,
            memory=memory,
            rotor_state=rotor_state,
            tokenizer=None,
            manual_quit=True,
        )

        matches = list(Path(tmpdir).glob("manual_quit_test_manual_quit_window*.pt"))
        assert len(matches) == 1
