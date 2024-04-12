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

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from model.KDSVD import KDSVDloss
from training.trainer_base import BaseTrainer, logger
from model.PKT import PKTloss
from model.RKD import RKDloss

from transformers.trainer_utils import speed_metrics
from transformers.trainer_pt_utils import nested_detach
from training.utils import AverageMeter, accuracy
from operator import itemgetter


def knowledgedistillation(inputs, inputs_t, T=4):
    # p_s = F.log_softmax(inputs.logits / T, dim=1)
    p_s = F.softmax(inputs.logits / T, dim=1)
    p_t = F.softmax(inputs_t.logits / T, dim=1)
    # print(p_s)
    # print(p_t)
    loss = F.kl_div(p_s, p_t, size_average=False) * (T ** 2) / inputs.logits.size()[0]
    return loss


def PKDLoss(s_features, t_features, num_hidden_layer_s, num_hidden_layer_t, hidden_size_s, hidden_size_t,
            max_seq_length):
    t_features = torch.cat(t_features[1:-1], dim=0).view(num_hidden_layer_t - 1,
                                                         -1,
                                                         max_seq_length,
                                                         hidden_size_t)[:, :, 0]

    s_features = torch.cat(s_features[1:-1], dim=0).view(num_hidden_layer_s - 1,
                                                         -1,
                                                         max_seq_length,
                                                         hidden_size_s)[:, :, 0]
    order = list(range(num_hidden_layer_t - 1))
    order = torch.LongTensor(order[-(num_hidden_layer_s - 1):])
    order, _ = order[:(num_hidden_layer_s - 1)].sort()

    t_features = itemgetter(order)(t_features)
    t_features = t_features / t_features.norm(dim=-1).unsqueeze(-1)
    s_features = s_features / s_features.norm(dim=-1).unsqueeze(-1)
    pkd_loss = F.mse_loss(s_features, t_features, reduction="mean")
    return pkd_loss


mse = torch.nn.MSELoss()
output_acc_epoch = [10, 20, 30, 40, 50, 60]
output_logit_epoch = [90]
test_arr = [i for i in range(10)]


class KDTrainer(BaseTrainer):
    def __init__(self, pkt=False, rkd=False, kdsvd=False, simkd=False, two_tokenizer=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pkt = pkt
        self.rkd = rkd
        self.kdsvd = kdsvd
        self.simkd = simkd
        self.two_tokenizer = two_tokenizer

    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    def training_step(self, model: nn.Module, inputs) -> torch.Tensor:
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
        model_t = self.teacher

        if self.two_tokenizer:
            input_student = {'input_ids': inputs['input_ids'].cuda(), 'attention_mask': inputs['attention_mask'].cuda(),
                             'labels': inputs['labels'].cuda(), 'token_type_ids': inputs['token_type_ids'].cuda()}

            input_teacher = {'input_ids': inputs['inputs_ids2'].cuda(),
                             'attention_mask': inputs['attention_mask2'].cuda(),
                             'labels': inputs['labels'].cuda()}

            with self.compute_loss_context_manager():
                loss = self.compute_loss_(model, model_t, input_student, input_teacher, alpha=0.4)
        else:
            inputs = self._prepare_inputs(inputs)
            inputs = {'input_ids': inputs['input_ids'].cuda(), 'attention_mask': inputs['attention_mask'].cuda(),
                      'labels': inputs['labels'].cuda()}
            with self.compute_loss_context_manager():
                loss = self.compute_loss_(model, model_t, inputs, alpha=0.3)

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

    def compute_loss_(self, model: nn.Module, model_t: nn.Module, input_s, input_t,
                      alpha=0.4, beta=0, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in input_s:
            labels = input_s.pop("labels")
        else:
            labels = None
        with torch.no_grad():
            teacher_outputs1 = model_t(**input_t)

        outputs = model(**input_s)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        kd_loss = knowledgedistillation(outputs, teacher_outputs1)

        if len(outputs.logits.shape) > 2:
            # print(teacher_outputs1.logits)
            pkd_loss = PKDLoss(s_features=outputs.logits, t_features=teacher_outputs1.logits,
                               num_hidden_layer_s=12, num_hidden_layer_t=24,
                               hidden_size_s=768, hidden_size_t=1024, max_seq_length=128)
        else:
            pkd_loss = 0

        loss = loss * (1 - alpha) + kd_loss * alpha + beta * pkd_loss

        return (loss, outputs) if return_outputs else loss

    def compute_loss_(self, model: nn.Module, model_t: nn.Module, inputs, alpha=0.5, beta=100, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        with torch.no_grad():
            teacher_outputs1 = model_t(**inputs)

        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        kd_loss = mse(outputs.logits, teacher_outputs1.logits)

        if len(outputs.logits.shape) > 2:
            print(teacher_outputs1.logits)
            pkd_loss = PKDLoss(s_features=outputs.logits, t_features=teacher_outputs1.logits,
                               num_hidden_layer_s=12, num_hidden_layer_t=24,
                               hidden_size_s=768, hidden_size_t=1024, max_seq_length=128)
        else:
            pkd_loss = 0

        loss = loss * (1 - alpha) + kd_loss * alpha + beta * pkd_loss

        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model: nn.Module, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval, model_t=None):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        eval_metrics = None
        if self.control.should_evaluate:
            eval_metrics, pred = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, eval_metrics)

            if epoch in output_acc_epoch:
                print('epoch: ', epoch, ' best accuracy: ', self.best_metrics["best_eval_" + self.test_key])

            if eval_metrics["eval_" + self.test_key] > self.best_metrics["best_eval_" + self.test_key]:
                with open('logits.txt', 'w') as fp:
                    fp.writelines(str(pred))

                self.best_metrics["best_epoch"] = epoch
                self.best_metrics["best_eval_" + self.test_key] = eval_metrics["eval_" + self.test_key]

                if self.predict_dataset is not None:
                    if isinstance(self.predict_dataset, dict):
                        for dataset_name, dataset in self.predict_dataset.items():
                            _, _, test_metrics = self.predict(dataset, metric_key_prefix="test")
                            self.best_metrics[f"best_test_{dataset_name}_{self.test_key}"] = test_metrics[
                                "test_" + self.test_key]
                    else:
                        _, _, test_metrics = self.predict(self.predict_dataset, metric_key_prefix="test")
                        self.best_metrics["best_test_" + self.test_key] = test_metrics["test_" + self.test_key]

            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=eval_metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        return dataset

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = False

        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        if self.two_tokenizer:
            inputs = {'input_ids': inputs['input_ids'].cuda(), 'attention_mask': inputs['attention_mask'].cuda(),
                      'labels': inputs['labels'].cuda(), 'token_type_ids': inputs['token_type_ids'].cuda()}
        else:
            inputs = self._prepare_inputs(inputs)
            inputs = {'input_ids': inputs['input_ids'].cuda(), 'attention_mask': inputs['attention_mask'].cuda(),
                      'labels': inputs['labels'].cuda()}

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

            logits = outputs[1:]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Tuple[Any, Any]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics, output.predictions

