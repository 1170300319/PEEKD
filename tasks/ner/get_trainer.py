import logging
import os
import random
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from model.LR import LogisticRegression
from tasks.ner.dataset import NERDataset
from training.trainer_exp import ExponentialTrainer
from model.utils import get_model, TaskType
from tasks.utils import ADD_PREFIX_SPACE, USE_FAST
from training.trainer_exp_multi_kd import ExpMultiKDTrainer
from training.trainer_expft import ExponentialFTTrainer
from training.trainer_expkd import ExponentialKDTrainer
from training.trainer_kd import KDTrainer
from torch.optim import Adam

logger = logging.getLogger(__name__)


def get_trainer(args):
    model_args, data_args, training_args, qa_args = args
    training_args.dataset_name = data_args.dataset_name

    training_args.last_tokenizer = data_args.last_tokenizer

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    model_type = AutoConfig.from_pretrained(model_args.model_name_or_path).model_type

    add_prefix_space = ADD_PREFIX_SPACE[model_type]

    use_fast = USE_FAST[model_type]

    type_s = 1 if 'bert' in model_args.model_name_or_path else 2
    if data_args.kd:
        type_t = 1 if 'bert' in model_args.teacher_name_or_path else 2
    else:
        type_t = type_s

    print(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=use_fast,
        revision=model_args.model_revision,
        add_prefix_space=add_prefix_space,
    )

    tokenizer_plus = None
    dataset = NERDataset(tokenizer=tokenizer, data_args=data_args, training_args=training_args,
                               tokenizer_plus=tokenizer_plus)

    print('kd ', data_args.kd)
    print('ft ', data_args.ft)
    print('moe ', data_args.moe)

    if training_args.do_train:
        for index in random.sample(range(len(dataset.train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {dataset.train_dataset[index]}.")

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=dataset.num_labels,
        label2id=dataset.label_to_id,
        id2label={i: l for l, i in dataset.label_to_id.items()},
        revision=model_args.model_revision,
    )
    if data_args.kd:
        teacher_config = AutoConfig.from_pretrained(
            model_args.teacher_name_or_path,
            num_labels=dataset.num_labels,
            label2id=dataset.label_to_id,
            id2label={i: l for l, i in dataset.label_to_id.items()},
            revision=model_args.model_revision,
        )
        if data_args.multiteacher:
            teacher_2_config = AutoConfig.from_pretrained(
                model_args.teacher_2_name_or_path,
                num_labels=dataset.num_labels,
                label2id=dataset.label_to_id,
                id2label={i: l for l, i in dataset.label_to_id.items()},
                revision=model_args.model_revision,
            )

    else:
        teacher_config = None

    if data_args.kd:
        model = get_model(model_args, TaskType.TOKEN_CLASSIFICATION,
                          config, model_name_or_path=model_args.model_name_or_path, fix_bert=True)
        if data_args.moe:
            teacher = get_model(model_args, TaskType.TOKEN_CLASSIFICATION_MOE,
                                teacher_config, model_name_or_path=model_args.teacher_name_or_path, fix_bert=True,
                                get_teacher=True)
        else:
            teacher = get_model(model_args, TaskType.TOKEN_CLASSIFICATION,
                                teacher_config, model_name_or_path=model_args.teacher_name_or_path, fix_bert=True,
                                get_teacher=True)
            if data_args.multiteacher:
                teacher2 = get_model(model_args, TaskType.TOKEN_CLASSIFICATION, teacher_2_config,
                                     model_name_or_path=model_args.teacher_2_name_or_path, fix_bert=True,
                                     get_teacher=True, psl=model_args.pre_seq_len_t2,
                                     prefix_projection=model_args.prefix_projection2)

    else:
        if data_args.moe:
            model = get_model(model_args, TaskType.TOKEN_CLASSIFICATION_MOE,
                              config, model_name_or_path=model_args.model_name_or_path, fix_bert=True)
        else:
            model = get_model(model_args, TaskType.TOKEN_CLASSIFICATION,
                              config, model_name_or_path=model_args.model_name_or_path, fix_bert=True)
        teacher = None

    model = model.cuda()
    if data_args.kd:
        teacher = teacher.cuda()
        if data_args.multiteacher:
            teacher2 = teacher2.cuda()

    if data_args.kd:
        if data_args.multiteacher:
            if data_args.lr_ensemble:
                weighter = LogisticRegression(n=2)
                opt = Adam(weighter.parameters(), lr=0.1)

            trainer = ExpMultiKDTrainer(
                model=model,
                teacher_list=[teacher, teacher2],
                args=training_args,
                train_dataset=dataset.train_dataset if training_args.do_train else None,
                eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
                compute_metrics=dataset.compute_metrics,
                tokenizer=tokenizer,
                data_collator=dataset.data_collator,
                test_key="f1",
                single_teacher=data_args.single_teacher,
                u_ensemble=data_args.u_ensemble,
                rand_single_ensemble=data_args.rand_single_ensemble,
                w_ensemble=data_args.w_ensemble,
                lr_ensemble=data_args.lr_ensemble,
                best_single=data_args.best_single,
                weighter=None if not data_args.lr_ensemble else (weighter, opt),
            )
        else:
            trainer = ExponentialKDTrainer(
                model=model,
                teacher=teacher,
                args=training_args,
                train_dataset=dataset.train_dataset if training_args.do_train else None,
                eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
                compute_metrics=dataset.compute_metrics,
                tokenizer=tokenizer,
                data_collator=dataset.data_collator,
                test_key="f1"
            )
    elif data_args.ft:
        # trainer = ExponentialFTTrainer(
        trainer = ExponentialTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset.train_dataset if training_args.do_train else None,
            eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
            compute_metrics=dataset.compute_metrics,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
            test_key="f1"
        )
    else:
        trainer = ExponentialTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset.train_dataset if training_args.do_train else None,
            eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
            predict_dataset=dataset.predict_dataset if training_args.do_predict else None,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
            compute_metrics=dataset.compute_metrics,
            test_key="f1"
        )
    return trainer, dataset.predict_dataset
