import logging
import os
import random
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from model.LR import LogisticRegression
from model.utils import get_model, TaskType, labels_to_prefix_tokens, task_to_lable
from tasks.superglue.dataset import SuperGlueDataset
from training.trainer_base import BaseTrainer
from training.trainer_exp import ExponentialTrainer
from training.trainer_qa import QuestionAnsweringTrainer
from training.trainer_kd import KDTrainer
from training.trainer_multi_kd import MultiKDTrainer
from training.trainer_ft import FTTrainer
import torch.nn as nn
from torch.optim import Adam

from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt.pipeline_base import PromptForClassification
from openprompt import PromptDataLoader

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def get_trainer(args):
    model_args, data_args, training_args, _ = args
    #print('model args, ', model_args.pre_seq_len)
    training_args.first_tokenizer = data_args.first_tokenizer
    training_args.second_tokenizer = data_args.second_tokenizer
    training_args.last_tokenizer = data_args.last_tokenizer
    training_args.dataset_name = data_args.dataset_name

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    type_s = 1 if 'bert' in model_args.model_name_or_path else 2
    if data_args.kd:
        type_t = 1 if 'bert' in model_args.teacher_name_or_path else 2
    else:
        type_t = type_s

    print(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    if type_s != type_t:
        tokenizer_plus = AutoTokenizer.from_pretrained(
            model_args.teacher_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
        )
    elif data_args.multiteacher:
        tokenizer_plus = AutoTokenizer.from_pretrained(
            model_args.teacher_2_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
        )
    else:
        tokenizer_plus = None

    print('tokenizer plus, ', tokenizer_plus)
    dataset = SuperGlueDataset(tokenizer=tokenizer, data_args=data_args, training_args=training_args,
                               tokenizer_plus=tokenizer_plus)

    if training_args.do_train:
        for index in random.sample(range(len(dataset.train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {dataset.train_dataset[index]}.")

    print('kd ', data_args.kd)
    print('ft ', data_args.ft)
    print('moe ', data_args.moe)

    if not dataset.multiple_choice:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )
        if data_args.kd:
            teacher_config = AutoConfig.from_pretrained(
                model_args.teacher_name_or_path,
                num_labels=dataset.num_labels,
                label2id=dataset.label2id,
                id2label=dataset.id2label,
                finetuning_task=data_args.dataset_name,
                revision=model_args.model_revision,
            )
            if data_args.multiteacher:
                teacher_2_config = AutoConfig.from_pretrained(
                    model_args.teacher_2_name_or_path,
                    num_labels=dataset.num_labels,
                    label2id=dataset.label2id,
                    id2label=dataset.id2label,
                    finetuning_task=data_args.dataset_name,
                    revision=model_args.model_revision,
                )
                teacher_3_config = AutoConfig.from_pretrained(
                    model_args.teacher_3_name_or_path,
                    num_labels=dataset.num_labels,
                    label2id=dataset.label2id,
                    id2label=dataset.id2label,
                    finetuning_task=data_args.dataset_name,
                    revision=model_args.model_revision,
                )

        else:
            teacher_config = None
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )
        if data_args.kd:
            teacher_config = AutoConfig.from_pretrained(
                model_args.teacher_name_or_path,
                num_labels=dataset.num_labels,
                finetuning_task=data_args.dataset_name,
                revision=model_args.model_revision,
            )
            if data_args.multiteacher:
                teacher_2_config = AutoConfig.from_pretrained(
                    model_args.teacher_2_name_or_path,
                    num_labels=dataset.num_labels,
                    finetuning_task=data_args.dataset_name,
                    revision=model_args.model_revision,
                )
                teacher_3_config = AutoConfig.from_pretrained(
                    model_args.teacher_3_name_or_path,
                    num_labels=dataset.num_labels,
                    finetuning_task=data_args.dataset_name,
                    revision=model_args.model_revision,
                )

        else:
            teacher_config = None


    if data_args.simkd:
        config.use_projector = True
    else:
        config.use_projector = False
    if teacher_config is not None:
        teacher_config.use_projector = False

    if not dataset.multiple_choice:
        if data_args.base:
            config.num_hidden_layers = 6  # 6

        if data_args.kd:
            if data_args.prototype:
                model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION_PROTOTYPE, config,
                                  model_name_or_path=model_args.model_name_or_path, get_teacher=False,
                                  prefix_projection=model_args.prefix_projection)
                teacher = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION_PROTOTYPE, teacher_config,
                                    model_name_or_path=model_args.teacher_name_or_path, fix_bert=True, get_teacher=True)
            else:
                model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config,
                                  model_name_or_path=model_args.model_name_or_path, get_teacher=False,
                                  prefix_projection=model_args.prefix_projection)
                if data_args.moe:
                    teacher = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION_MOE, teacher_config,
                                        model_name_or_path=model_args.teacher_name_or_path, fix_bert=True, get_teacher=True)
                else:
                    teacher = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, teacher_config,
                                        model_name_or_path=model_args.teacher_name_or_path, fix_bert=True, get_teacher=True)
                    if data_args.multiteacher:
                        teacher2 = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, teacher_2_config,
                                            model_name_or_path=model_args.teacher_2_name_or_path, fix_bert=True,
                                            get_teacher=True, psl=model_args.pre_seq_len_t2,
                                             prefix_projection=model_args.prefix_projection2)
                        teacher3 = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, teacher_3_config,
                                            model_name_or_path=model_args.teacher_3_name_or_path, fix_bert=True,
                                            get_teacher=True, psl=model_args.pre_seq_len_t3)
        else:
            if data_args.prototype:
                model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION_PROTOTYPE, config,
                                  model_name_or_path=model_args.model_name_or_path, get_teacher=False,
                                  prefix_projection=model_args.prefix_projection)
            elif data_args.moe:
                model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION_MOE, config,
                                  model_name_or_path=model_args.model_name_or_path, get_teacher=False,
                                  prefix_projection=model_args.prefix_projection)
            else:
                model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config,
                                  model_name_or_path=model_args.model_name_or_path, get_teacher=False,
                                  prefix_projection=model_args.prefix_projection)
            teacher = None
    else:
        if data_args.base:
            config.num_hidden_layers = 6  # 6

        if data_args.kd:
            if data_args.moe:
                model = get_model(model_args, TaskType.MULTIPLE_CHOICE, config,
                                  model_name_or_path=model_args.model_name_or_path,
                                  fix_bert=True, get_teacher=False)
                teacher = get_model(model_args, TaskType.MULTIPLE_CHOICE_MOE, teacher_config,
                                    model_name_or_path=model_args.teacher_name_or_path,
                                    fix_bert=True, get_teacher=True)
            else:
                model = get_model(model_args, TaskType.MULTIPLE_CHOICE, config,
                                    model_name_or_path=model_args.model_name_or_path,
                                    fix_bert=True, get_teacher=False)
                teacher = get_model(model_args, TaskType.MULTIPLE_CHOICE, teacher_config,
                                    model_name_or_path=model_args.teacher_name_or_path,
                                    fix_bert=True, get_teacher=True)
                if data_args.multiteacher:
                    teacher2 = get_model(model_args, TaskType.MULTIPLE_CHOICE, teacher_2_config,
                                         model_name_or_path=model_args.teacher_2_name_or_path, fix_bert=True,
                                         get_teacher=True, psl=model_args.pre_seq_len_t2)
                    teacher3 = get_model(model_args, TaskType.MULTIPLE_CHOICE, teacher_3_config,
                                         model_name_or_path=model_args.teacher_3_name_or_path, fix_bert=True,
                                         get_teacher=True, psl=model_args.pre_seq_len_t3)

        else:
            if data_args.moe:
                model = get_model(model_args, TaskType.MULTIPLE_CHOICE_MOE, config,
                                  model_name_or_path=model_args.model_name_or_path,
                                  fix_bert=True, get_teacher=False)
            else:
                model = get_model(model_args, TaskType.MULTIPLE_CHOICE, config,
                                  model_name_or_path=model_args.model_name_or_path,
                                  fix_bert=True, get_teacher=False)
            teacher = None

    print('model to cuda')
    model = model.cuda()
    if data_args.kd:
        teacher = teacher.cuda()
        if data_args.multiteacher:
            teacher2 = teacher2.cuda()
            teacher3 = teacher3.cuda()

    print('Init trainer')
    # Initialize our Trainer
    if data_args.kd:
        if data_args.multiteacher:
            if data_args.lr_ensemble:
                weighter = LogisticRegression(n=3)
                opt = Adam(weighter.parameters(), lr=0.001)

            trainer = MultiKDTrainer(
                model=model,
                teacher_list=[teacher, teacher2, teacher3],
                args=training_args,
                train_dataset=dataset.train_dataset if training_args.do_train else None,
                eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
                compute_metrics=dataset.compute_metrics,
                tokenizer=tokenizer,
                data_collator=dataset.data_collator,
                test_key=dataset.test_key,
                single_teacher=data_args.single_teacher,
                u_ensemble=data_args.u_ensemble,
                rand_single_ensemble=data_args.rand_single_ensemble,
                w_ensemble=data_args.w_ensemble,
                lr_ensemble=data_args.lr_ensemble,
                best_single=data_args.best_single,
                weighter=None if not data_args.lr_ensemble else (weighter, opt),
            )
        else:
            trainer = KDTrainer(
                model=model,
                teacher=teacher,
                args=training_args,
                train_dataset=dataset.train_dataset if training_args.do_train else None,
                eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
                compute_metrics=dataset.compute_metrics,
                tokenizer=tokenizer,
                data_collator=dataset.data_collator,
                test_key=dataset.test_key,
                pkt=True if data_args.pkt else False,
                rkd=True if data_args.rkd else False,
                kdsvd=True if data_args.kdsvd else False,
                simkd=True if data_args.simkd else False,
            )
    elif data_args.ft:
        trainer = FTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset.train_dataset if training_args.do_train else None,
            eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
            compute_metrics=dataset.compute_metrics,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
            test_key=dataset.test_key
        )
    else:
        trainer = BaseTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset.train_dataset if training_args.do_train else None,
            eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
            compute_metrics=dataset.compute_metrics,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
            test_key=dataset.test_key
        )

    return trainer, None


