import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.models.wav2vec2.components as components
from dataset import (
    BucketizeBatchSampler,
    CollateFnHubert,
    DistributedBatchSampler,
    HuBERTDataSet,
)
from lightning.pytorch import LightningModule
from loss import hubert_loss
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


Batch = Tuple[Tensor, Tensor, Tensor]


class LinearDecayLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear learning rate scheduler with warm up."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_updates: int,
        max_updates: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_updates = warmup_updates
        self.max_updates = max_updates
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            return [self._step_count / self.warmup_updates * base_lr for base_lr in self.base_lrs]
        elif self._step_count >= self.max_updates:
            return [0.0 for _ in self.base_lrs]
        else:
            pct_remaining = (self.max_updates - self._step_count) / (self.max_updates - self.warmup_updates)
            return [base_lr * pct_remaining for base_lr in self.base_lrs]




def _compute_accuracy(logits: torch.Tensor):
    with torch.no_grad():
        max = logits.argmax(-1) == 0
        min = logits.argmin(-1) == 0
        both = max & min
        corr = max.long().sum().item() - both.long().sum().item()
        count = max.numel()
    return corr, count


def _reset_stats():
    return {
        "train": {
            "correct": 0.0,
            "count": 0.0,
        },
        "val": {
            "correct": 0.0,
            "count": 0.0,
        },
    }


class HuBERTPreTrainModule(LightningModule):
    # Hyperparameters - modify these for experimentation
    MASK_PROB = 0.4          # 40% of frames masked during training
    MASK_LENGTH = 5          # 5 frames = 102ms mask duration
    MASK_CHANNEL_LENGTH = 5  # 5 channels masked

    def __init__(
        self,
        *,
        model_name: str,
        feature_grad_mult: float,
        num_classes: int,
        dataset: str,
        dataset_path: str,
        feature_type: str,
        seconds_per_batch: float,
        learning_rate: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
        clip_norm: Optional[float],
        warmup_updates: int,
        max_updates: int,
        kernel_size_ms: int = 25,
        stride_ms: int = 20,
        sample_rate: int = 16000,
    ):
        super().__init__()

        if model_name == "hubert_pretrain_base":
            self.model = torchaudio.models.hubert_pretrain_base(
                feature_grad_mult=feature_grad_mult,
                num_classes=num_classes,
                mask_prob=self.MASK_PROB,
                mask_length=self.MASK_LENGTH,
                mask_channel_length=self.MASK_CHANNEL_LENGTH,
            )
        elif model_name == "hubert_pretrain_large":
            self.model = torchaudio.models.hubert_pretrain_large()
        elif model_name == "hubert_pretrain_xlarge":
            self.model = torchaudio.models.hubert_pretrain_xlarge()
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        self.automatic_optimization = False
        self.scaler = torch.cuda.amp.GradScaler()

        self.loss = hubert_loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay
        )
        self.clip_norm = clip_norm
        self.lr_scheduler = LinearDecayLRScheduler(self.optimizer, warmup_updates, max_updates)
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.feature_type = feature_type
        self.seconds_per_batch = seconds_per_batch
        self.kernel_size_ms = kernel_size_ms
        self.stride_ms = stride_ms
        self.sample_rate = sample_rate
        self.mask_stats = _reset_stats()
        self.unmask_stats = _reset_stats()
        self.nan_loss_count = 0.0

    def _step(self, batch: Batch, batch_idx, step_type):
        if batch is None:
            return None, None
        waveforms, labels, audio_lengths = batch
        if step_type == "val":
            with torch.no_grad():
                logit_m, logit_u, feature_penalty = self.model(
                    waveforms,
                    labels,
                    audio_lengths,
                )
        else:
            logit_m, logit_u, feature_penalty = self.model(
                waveforms,
                labels,
                audio_lengths,
            )
        loss = self.loss(logit_m, logit_u, feature_penalty)
        if not torch.isinf(loss) and not torch.isnan(loss):
            self.log(f"{step_type}_loss", loss.item() / logit_m.size(0), on_step=True, on_epoch=True)
        else:
            self.nan_loss_count += 1
            self.log("nan_loss_count", self.nan_loss_count, on_step=True, on_epoch=True)

        # log accuracies of masked and unmasked frames
        correct_m, count_m = _compute_accuracy(logit_m)
        correct_u, count_u = _compute_accuracy(logit_u)
        self.mask_stats[step_type]["correct"] += correct_m
        self.mask_stats[step_type]["count"] += count_m
        self.unmask_stats[step_type]["correct"] += correct_u
        self.unmask_stats[step_type]["count"] += count_u
        self.log(
            f"{step_type}_masked_accuracy",
            self.mask_stats[step_type]["correct"] / self.mask_stats[step_type]["count"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=step_type == "train",
        )
        self.log(
            f"{step_type}_unmasked_accuracy",
            self.unmask_stats[step_type]["correct"] / self.unmask_stats[step_type]["count"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=step_type == "train",
        )
        return loss, logit_m.size(0)

    def configure_optimizers(self):
        return (
            [self.optimizer],
            [
                {
                    "scheduler": self.lr_scheduler,
                    "interval": "step",
                },
            ],
        )

    def training_step(self, batch: Batch, batch_idx):
        """Custom training step with loss normalization and automatic mixed precision training.

        By default, DDP does the following on each train step:
        - For each GPU, compute loss and gradient on shard of training data.
        - Sync and average gradients across all GPUs. The final gradient
          is (sum of gradients across all GPUs) / N, where N is the world
          size (total number of GPUs).
        - Update parameters on each GPU.

        Here, we do the following:
        - For k-th GPU, compute loss and scale it by (N / num_frames), where num_frames is
          the sum of masked frames across all GPUs. Compute gradient from scaled loss.
        - Sync and average gradients across all GPUs. The final gradient
          is (sum of gradients across all GPUs) / num_frames.
        - Update parameters on each GPU.

        Doing so allows us to account for the variability in number of masked frames in
        variable-length sequential data.
        """
        opt = self.optimizers()
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            loss, num_frame = self._step(batch, batch_idx, "train")
        if torch.isinf(loss) or torch.isnan(loss):
            opt.zero_grad()
            return None

        # normalize the loss based on the sum of num_frame across all GPUs
        num_frames = self.all_gather(num_frame)
        self.log("Gathered number of frames", num_frames.float().sum(), on_step=True, on_epoch=True)
        loss *= num_frames.size(0) / num_frames.sum()  # world size / num_frames

        # backward the loss and clip the gradients
        loss = self.scaler.scale(loss)
        self.manual_backward(loss)
        self.scaler.unscale_(opt)
        if self.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        # optimization
        self.scaler.step(opt)
        sch = self.lr_schedulers()
        sch.step()
        self.scaler.update()
        return loss

    def validation_step(self, batch: Batch, batch_idx):
        return self._step(batch, batch_idx, "val")[0]

    def on_validation_end(self):
        self.mask_stats = _reset_stats()
        self.unmask_stats = _reset_stats()

    def train_dataloader(self):
        dataset = HuBERTDataSet(self.dataset_path, self.dataset, "train")
        sampler = BucketizeBatchSampler(
            dataset.len_list,
            num_buckets=1000,
            max_token_count=self.seconds_per_batch * 16000,
            min_len=32000,
            max_len=250000,
            shuffle=True,
            seed=self.trainer.current_epoch,
        )
        sampler = DistributedBatchSampler(sampler, shuffle=True)
        sampler.set_epoch(self.current_epoch)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnHubert(
                feature_type=self.feature_type,
                pad=False,
                rand_crop=True,
                kernel_size_ms=self.kernel_size_ms,
                stride_ms=self.stride_ms,
                sample_rate=self.sample_rate,
            ),
            num_workers=10,
        )
        return dataloader

    #def val_dataloader(self):
        #dataset = HuBERTDataSet(self.dataset_path, self.dataset, "valid")
        #sampler = BucketizeBatchSampler(
        #    dataset.len_list,
        #    num_buckets=1000,
        #    max_token_count=self.seconds_per_batch * 16000,
        #    min_len=32000,
        #    max_len=250000,
        #    shuffle=False,
        #)
        #dataloader = DataLoader(
        #    dataset,
        #    batch_sampler=sampler,
        #    collate_fn=CollateFnHubert(
        #        feature_type=self.feature_type,
        #        pad=False,
        #        rand_crop=True,
        #        kernel_size_ms=self.kernel_size_ms,
        #        stride_ms=self.stride_ms,
        #        sample_rate=self.sample_rate,
        #    ),
        #    num_workers=10,
        #)
        #return None
        # return dataloader # disable validation for pretraining