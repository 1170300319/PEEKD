import logging
import os
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import Trainer
import torch

logger = logging.getLogger(__name__)

_default_log_level = logging.INFO
logger.setLevel(_default_log_level)


class BaseTrainer(Trainer):
    def __init__(self, *args, predict_dataset=None, test_key="accuracy", teacher=None, **kwargs, ):
        super().__init__(*args, **kwargs)
        self.predict_dataset = predict_dataset
        self.teacher = teacher
        self.test_key = test_key
        # 改这里加metrics
        self.best_metrics = OrderedDict({
            "kl_loss": 0,
            "agree": 0,
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })

    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)

    def evaluateasentence(self, idx=1, info=False):
        inputs = self.eval_dataset[idx]
        if info:
            print('premise: ', self.tokenizer(inputs['premise']))
            #print('premise: ', self.tokenizer.tokenize(inputs['premise']))
            #print('hypothesis: ', self.tokenizer.tokenize(inputs['hypothesis']))
        inputs = self._prepare_inputs(inputs)
        prompt = 'Premise and hypothesis are as follows.'
        #prompt = 'Judge the relation between following premise and hypothesis.'
        prompt_idx = self.tokenizer(prompt).input_ids
        if info:
            print(prompt_idx)
            print(inputs['input_ids'][:128 - len(prompt_idx)])
            print('input: ', inputs)
        inputs = {'input_ids': torch.tensor([np.append(prompt_idx, inputs['input_ids'][:128-len(prompt_idx)])]).cuda(),
                  'attention_mask': torch.tensor([np.append(np.ones(len(prompt_idx)),
                                                            inputs['attention_mask'][:128-len(prompt_idx)])]).cuda()}
        model = self.model
        output = model(**inputs, output_attentions=True)
        if info:
            print('data: ', inputs)
            print(output.logits)
        return output.logits

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
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

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            use_prompt=False
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
            :param use_prompt:
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

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

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics, output.predictions
