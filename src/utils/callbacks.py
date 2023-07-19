"""Collection of custom callbacks for PyTorch Lightning."""
import signal
import socket
from pathlib import Path

import pytorch_lightning.callbacks as plc

__all__ = ['TQDMProgressBar', 'RichProgressBar', 'DetectAnomaly', 'TrainingManager']


class TQDMProgressBar(plc.TQDMProgressBar):
    """Progress bar that removes all `grad norms` from display."""
    def get_metrics(self, trainer, pl_module) -> dict:
        m = super().get_metrics(trainer, pl_module)
        m = {k: v for k, v in m.items() if 'grad' not in k}
        return m


class RichProgressBar(plc.RichProgressBar):
    """Progress bar that removes all `grad norms` from display."""
    def get_metrics(self, trainer, pl_module) -> dict:
        m = super().get_metrics(trainer, pl_module)
        m = {k: v for k, v in m.items() if 'grad' not in k}
        return m


class DetectAnomaly(plc.Callback):
    """Check for NaN/infinite loss at each core step. Replacement for `detect_anomaly=True`."""
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0) -> None:
        if not (loss := outputs['loss']).isfinite():
            raise ValueError(f'Detected NaN/Infinite loss: "{loss}"')


class TrainingManager(plc.Callback):
    """Callback to save a dummy file as an indicator when training has started/finished."""
    # FIXME: Unsure if there are edge cases where `training` is not deleted on HPC clusters...
    def __init__(self, ckpt_dir: Path):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        self.host = socket.gethostname()

        self.ftrain = None
        self.fend = ckpt_dir/'finished'  # Finished placeholder

        # NOTE: Since exception happens in `__init__` we won't call `self.on_exception`.
        # This is the desired behaviour!
        if self.is_training: raise ValueError(f'Training already in progress! ({self.ftrain})')
        if self.has_finished: raise ValueError(f'Training already finished! ({self.fend})')

        signal.signal(signal.SIGTERM, self._on_sigterm)

    @property
    def is_training(self) -> bool:
        fs = sorted(self.ckpt_dir.glob('training*'))
        n = len(fs)
        if n == 0: return False
        if n == 1:
            self.ftrain = fs[0]
            return True
        raise ValueError(f'Invalid number of training files! {fs}')

    @property
    def has_finished(self) -> bool: return self.fend.is_file()

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        print(f'-> Creating "training" file...')
        if self.ftrain: self.ftrain.unlink(missing_ok=True)
        self.ftrain = self.ckpt_dir / f'training_{trainer.current_epoch}_{self.host}'
        self.ftrain.touch()

    def on_fit_end(self, trainer, pl_module) -> None:
        self._cleanup()
        print('-> Creating "finished"" file...')
        self.fend.touch()

    def on_exception(self, trainer, pl_module, exception) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        print('-> Deleting "training" file...')
        if self.ftrain: self.ftrain.unlink(missing_ok=True)
        print('-> Done! Exiting...')

    def _on_sigterm(self, signum, frame) -> None:
        """Signature required by `signal.signal`."""
        raise SystemExit  # Ensure we call `self._cleanup`.
