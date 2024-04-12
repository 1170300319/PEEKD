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

from transformers.trainer_pt_utils import nested_detach
from training.utils import AverageMeter, accuracy
from operator import itemgetter

from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss


def knowledgedistillation(inputs, inputs_t, T=4):
    p_s = F.softmax(inputs / T, dim=1)
    p_t = F.softmax(inputs_t / T, dim=1)
    loss = F.kl_div(p_s, p_t, size_average=False) * (T ** 2) / inputs.size()[0]
    return loss


mse = torch.nn.MSELoss()
output_acc_epoch = [10, 20, 30, 40, 50, 60]


class MultiKDTrainer(BaseTrainer):
    def __init__(self, single_teacher=False, u_ensemble=False, rand_single_ensemble=False,
                 w_ensemble=False, lr_ensemble=False, best_single=False, selector_single=False,
                 teacher_list=None, weighter=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.single_teacher = single_teacher
        self.u_ensemble = u_ensemble
        self.rand_single_ensemble = rand_single_ensemble
        self.w_ensemble = w_ensemble
        self.lr_ensemble = lr_ensemble
        self.best_single = best_single
        self.selector_single = selector_single
        self.teacher_list = teacher_list
        if weighter is not None:
            self.weighter = weighter[0]
            self.weighter_opt = weighter[1]

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
        model_ts = self.teacher_list
        for model_t in model_ts:
            model_t.eval()
        input_student = {'input_ids': inputs['input_ids'].cuda(), 'attention_mask': inputs['attention_mask'].cuda(),
                         'labels': inputs['labels'].cuda()}
        input_teacher = []
        if self.args.first_tokenizer == 1:
            input_teacher.append({'input_ids': inputs['input_ids'].cuda(),
                                  'attention_mask': inputs['attention_mask'].cuda(), 'labels': inputs['labels'].cuda()})
        else:
            input_teacher.append({'input_ids': inputs['input_ids2'].cuda(),
                                  'attention_mask': inputs['attention_mask2'].cuda(),
                                  'labels': inputs['labels'].cuda(),
                                  'token_type_ids': inputs['token_type_ids'].cuda()} if 'token_type_ids' in inputs else \
                                     {'input_ids': inputs['input_ids2'].cuda(),
                                      'attention_mask': inputs['attention_mask2'].cuda(),
                                      'labels': inputs['labels'].cuda()})
        if self.args.second_tokenizer == 1:
            input_teacher.append({'input_ids': inputs['input_ids'].cuda(),
                                  'attention_mask': inputs['attention_mask'].cuda(), 'labels': inputs['labels'].cuda()})
        else:
            input_teacher.append({'input_ids': inputs['input_ids2'].cuda(),
                                  'attention_mask': inputs['attention_mask2'].cuda(),
                                  'labels': inputs['labels'].cuda(),
                                  'token_type_ids': inputs['token_type_ids'].cuda()} if 'token_type_ids' in inputs else \
                                     {'input_ids': inputs['input_ids2'].cuda(),
                                      'attention_mask': inputs['attention_mask2'].cuda(),
                                      'labels': inputs['labels'].cuda()})
        if self.args.last_tokenizer == 1:
            input_teacher.append({'input_ids': inputs['input_ids'].cuda(),
                                  'attention_mask': inputs['attention_mask'].cuda(), 'labels': inputs['labels'].cuda()})
        else:
            input_teacher.append({'input_ids': inputs['input_ids2'].cuda(),
                                  'attention_mask': inputs['attention_mask2'].cuda(),
                                  'labels': inputs['labels'].cuda(),
                                  'token_type_ids': inputs['token_type_ids'].cuda()} if 'token_type_ids' in inputs else \
                                     {'input_ids': inputs['input_ids2'].cuda(),
                                      'attention_mask': inputs['attention_mask2'].cuda(),
                                      'labels': inputs['labels'].cuda()})

        with self.compute_loss_context_manager():
            loss = self.compute_loss_(model, model_ts, input_student, input_teacher, alpha=0.5)

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
            if self.lr_ensemble:
                self.weighter_opt.zero_grad()
                self.weighter_opt.step()

        return loss.detach()

    def compute_loss_(self, model: nn.Module, model_ts: list, input_s, input_t,
                      alpha=0.4, beta=100, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in input_s:
            labels = input_s.pop("labels")
        else:
            labels = None
        with torch.no_grad():
            teacher_outputs1 = model_ts[0](**input_t[0])
            teacher_outputs2 = model_ts[1](**input_t[1])
            teacher_outputs3 = model_ts[2](**input_t[2])

        outputs = model(**input_s)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.rand_single_ensemble:
            seed = random.randint(1, 2)
            if seed == 0:
                kd_loss = knowledgedistillation(outputs.logits, teacher_outputs1.logits)
            elif seed == 1:
                kd_loss = knowledgedistillation(outputs.logits, teacher_outputs2.logits)
            else:
                kd_loss = knowledgedistillation(outputs.logits, teacher_outputs3.logits)
        elif self.u_ensemble:
            ensemble_logits = teacher_outputs1.logits * 0.33 + teacher_outputs2.logits * 0.33 \
                              + teacher_outputs3.logits * 0.33
            kd_loss = knowledgedistillation(outputs.logits, ensemble_logits)
        elif self.w_ensemble:
            # 瞎编一个？
            ensemble_logits = teacher_outputs1.logits * 0.4 + teacher_outputs2.logits * 0.3 \
                              + teacher_outputs3.logits * 0.3
            kd_loss = knowledgedistillation(outputs.logits, ensemble_logits)
        elif self.lr_ensemble:
            weights = self.weighter.get_weights()
            ensemble_logits = teacher_outputs1.logits * weights[0] + teacher_outputs2.logits * weights[1] \
                              + teacher_outputs3.logits * weights[2]
            kd_loss = knowledgedistillation(outputs.logits, ensemble_logits)
        elif self.best_single:
            loss_fct = CrossEntropyLoss()

            loss_t1 = loss_fct(teacher_outputs1.logits, input_s['labels']).mean()
            loss_t2 = loss_fct(teacher_outputs2.logits, input_s['labels']).mean()
            loss_t3 = loss_fct(teacher_outputs3.logits, input_s['labels']).mean()

            if loss_t1 <= loss_t2 and loss_t1 <= loss_t3:
                kd_loss = knowledgedistillation(outputs.logits, teacher_outputs1.logits)
            elif loss_t2 <= loss_t1 and loss_t2 <= loss_t3:
                kd_loss = knowledgedistillation(outputs.logits, teacher_outputs2.logits)
            else:
                kd_loss = knowledgedistillation(outputs.logits, teacher_outputs3.logits)
        elif self.selector_single:
            pass

        loss = loss * (1 - alpha) + kd_loss * alpha

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
            eval_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, eval_metrics)

            if epoch in output_acc_epoch:
                print('epoch: ', epoch, ' best accuracy: ', self.best_metrics["best_eval_" + self.test_key])

            if eval_metrics["eval_" + self.test_key] > self.best_metrics["best_eval_" + self.test_key]:
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
