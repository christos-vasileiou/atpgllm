"""Accumulation correction for TRL 0.26.1's generation-normalized losses."""

from pathlib import Path


class GenerationBatchLossMixin:
    """Average generation-batch token means over an optimizer update.

    TRL's DAPO/CISPO denominator covers one generation batch, which is split
    into ``steps_per_generation`` microbatches. The base trainer deliberately
    bypasses Transformers' accumulation division. Scale each contribution by
    generation_steps / accumulation_steps so accumulating several generation
    batches averages their objectives instead of summing them.

    This is an average of generation-batch token means, not token weighting
    over every generation batch combined. It preserves bounded rollout memory.
    The actual accumulation length handles a short final optimizer update.
    Evaluation has no accumulation and must not receive this correction.
    """

    def _compute_loss(self, model, inputs):
        loss = super()._compute_loss(model, inputs)
        if self.model.training and self.loss_type in ("dapo", "cispo"):
            accumulation = self.current_gradient_accumulation_steps
            factor = self.args.steps_per_generation / accumulation
            self._metrics["train"].setdefault("loss/accumulation_scale", []).append(factor)
            return loss * factor
        return loss

    def _save_checkpoint(self, model, trial):
        # A reused checkpoint directory must not advertise the previous save
        # as complete while model/optimizer files are being replaced.
        if self.accelerator.is_main_process:
            marker = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}" / "training_state_summary.json"
            marker.unlink(missing_ok=True)
        self.accelerator.wait_for_everyone()
        return super()._save_checkpoint(model, trial)

    def compute_liger_loss(self, *args, **kwargs):
        # The fused backend has a different normalization contract. Do not
        # silently run an uncorrected loss if it is enabled in a future config.
        if self.loss_type in ("dapo", "cispo"):
            raise ValueError("Generation-batch normalization requires use_liger_kernel=False")
        return super().compute_liger_loss(*args, **kwargs)
