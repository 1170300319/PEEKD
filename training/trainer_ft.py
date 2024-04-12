import logging
import os
import random
import sys

from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union
import math
import random
import time
import warnings
import collections

from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import (
    HPSearchBackend,
    ShardedDDPOption,
    TrainOutput,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    is_torch_tpu_available,
)
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from training.trainer_base import BaseTrainer, logger
from training.utils import get_uncertenty, get_weights



class FTTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss_(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def compute_loss_(self, model: nn.Module, inputs, return_outputs=False, moe=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if moe:
            outputs1 = model(**inputs, prefix_num=0)
            outputs2 = model(**inputs, prefix_num=1)
        else:
            outputs1 = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if moe:
                loss = self.label_smoother(outputs1, labels) + self.label_smoother(outputs2, labels)
            else:
                loss = self.label_smoother(outputs1, labels)
        else:
            if moe:
                u1 = get_uncertenty(outputs1, inputs['labels'])
                u2 = get_uncertenty(outputs2, inputs['labels'])
                a1, a2 = get_weights(u1, u2)
                loss = a1*outputs1["loss"] + a2*outputs2["loss"] if isinstance(outputs1, dict) else a1*outputs1[0] + a2*outputs2[0]
            else:
                loss = outputs1["loss"] if isinstance(outputs1, dict) else outputs1[0]
        return (loss, outputs1) if return_outputs else loss
