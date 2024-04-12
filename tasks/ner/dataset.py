import torch
from torch.utils import data
from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoConfig
import numpy as np


class NERDataset:
    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args, tokenizer_plus: AutoTokenizer = None) -> None:
        super().__init__()
        raw_datasets = load_dataset(f'tasks/ner/datasets/{data_args.dataset_name}.py')
        self.tokenizer = tokenizer

        if tokenizer_plus is not None:
            self.tokenizer_plus = tokenizer_plus
        else:
            self.tokenizer_plus = None


        if training_args.do_train:
            column_names = raw_datasets["train"].column_names
            features = raw_datasets["train"].features
        else:
            column_names = raw_datasets["validation"].column_names
            features = raw_datasets["validation"].features

        self.label_column_name = f"{data_args.task_name}_tags"
        self.label_list = features[self.label_column_name].feature.names
        self.label_to_id = {l: i for i, l in enumerate(self.label_list)}
        self.num_labels = len(self.label_list)

        if training_args.do_train:
            train_dataset = raw_datasets['train']
            if data_args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(data_args.max_train_samples))
            self.train_dataset = train_dataset.map(
                self.tokenize_and_align_labels,
                batched=True,
                load_from_cache_file=True,
                desc="Running tokenizer on train dataset",
            )
        if training_args.do_eval:
            eval_dataset = raw_datasets['validation']
            if data_args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
            self.eval_dataset = eval_dataset.map(
                self.tokenize_and_align_labels,
                batched=True,
                load_from_cache_file=True,
                desc="Running tokenizer on validation dataset",
            )
        if training_args.do_predict:
            predict_dataset = raw_datasets['test']
            if data_args.max_predict_samples is not None:
                predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
            self.predict_dataset = predict_dataset.map(
                self.tokenize_and_align_labels,
                batched=True,
                load_from_cache_file=True,
                desc="Running tokenizer on test dataset",
            )

        self.data_collator = DataCollatorForTokenClassification(self.tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

        self.metric = load_metric('seqeval')

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples['tokens'],
            padding=False,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        tokenized_inputs = {'input_ids': tokenized_inputs['input_ids'],
                            'attention_mask': tokenized_inputs['attention_mask']}
        #if 'token_type_ids' in tokenized_inputs:
        #    tokenized_inputs["token_type_ids"] = tokenized_inputs['token_type_ids']

        #if False:
        if self.tokenizer_plus is not None:
            # print(examples)
            tokenized_inputs2 = self.tokenizer_plus(
                examples['tokens'],
                padding=False,
                truncation=True,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
            )
            tokenized_inputs["input_ids2"] = tokenized_inputs2['input_ids']
            tokenized_inputs["attention_mask2"] = tokenized_inputs2['attention_mask']
            #if 'token_type_ids' in tokenized_inputs2:
            #    tokenized_inputs["token_type_ids"] = tokenized_inputs2['token_type_ids']

        labels = []
        for i, label in enumerate(examples[self.label_column_name]):
            word_ids = [None]
            for j, word in enumerate(examples['tokens'][i]):
                token = self.tokenizer.encode(word, add_special_tokens=False)
                # print(token)
                word_ids += [j] * len(token)
            word_ids += [None]
            
            # word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                    # label_ids.append(self.label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs


if __name__ == '__main__':
    raw_datasets = load_dataset(f'tasks/ner/datasets/conll2003.py')
    #print(raw_datasets['train'][:5])

    tokenizer1 = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, add_prefix_space=False)
    tokenizer2 = AutoTokenizer.from_pretrained('roberta-base', use_fast=True, add_prefix_space=True)

    def preprocess_function1(examples):
        tokenized_inputs1 = tokenizer1(
            examples['tokens'],
            padding=False,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        tokenized_inputs2 = tokenizer2(
            examples['tokens'],
            padding=False,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        tokenized_inputs1 = {'input_ids': tokenized_inputs1['input_ids'],
                             'attention_mask': tokenized_inputs1['attention_mask'],
                             "input_ids2": tokenized_inputs2['input_ids'],
                             "attention_mask2": tokenized_inputs2['attention_mask']}

        if 'token_type_ids' in tokenized_inputs2:
            tokenized_inputs1["token_type_ids"] = tokenized_inputs2['token_type_ids']

        label_column_name = f"{'ner'}_tags"
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = [None]
            for j, word in enumerate(examples['tokens'][i]):
                token = tokenizer1.encode(word, add_special_tokens=False)
                # print(token)
                word_ids += [j] * len(token)
            word_ids += [None]

            # word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                    # label_ids.append(self.label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs1["labels"] = labels
        return tokenized_inputs1


    raw_datasets = raw_datasets.map(
        preprocess_function1,
        batched=True
    )

    # print(raw_datasets.column_names)
    raw_datasets = raw_datasets.remove_columns(['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'])
    print(raw_datasets['train'][:5])
