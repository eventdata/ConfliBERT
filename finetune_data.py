from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed,
)
from datasets import Dataset
from example_based import example_based_accuracy, example_based_recall, example_based_precision, example_based_f1
from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import argparse
from sklearn import metrics
import pandas as pd
import numpy as np
import csv
import os
import json


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _tokenize_classification(df, tokenizer, max_seq_length, multilabel=False):
    encodings = tokenizer(df['text'].tolist(), truncation=True, padding='max_length', max_length=max_seq_length)
    labels = [list(map(float, l)) for l in df['labels']] if multilabel else df['labels'].tolist()
    data = {'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask'], 'labels': labels}
    if 'token_type_ids' in encodings:
        data['token_type_ids'] = encodings['token_type_ids']
    return Dataset.from_dict(data)


def _group_ner_by_sentence(df):
    return [(g['words'].tolist(), g['labels'].tolist()) for _, g in df.groupby('sentence_id', sort=False)]


def _tokenize_ner(df, tokenizer, max_seq_length, label2id):
    has_token_type = 'token_type_ids' in tokenizer.model_input_names
    all_input_ids, all_attention_masks, all_token_type_ids, all_labels = [], [], [], []

    for words, word_labels in _group_ner_by_sentence(df):
        enc = tokenizer(words, is_split_into_words=True, truncation=True, padding='max_length', max_length=max_seq_length)
        label_ids, prev = [], None
        for wid in enc.word_ids():
            if wid is None or wid == prev:
                label_ids.append(-100)
            else:
                label_ids.append(label2id.get(word_labels[wid], -100))
            prev = wid
        all_input_ids.append(enc['input_ids'])
        all_attention_masks.append(enc['attention_mask'])
        all_labels.append(label_ids)
        if has_token_type:
            all_token_type_ids.append(enc['token_type_ids'])

    data = {'input_ids': all_input_ids, 'attention_mask': all_attention_masks, 'labels': all_labels}
    if has_token_type:
        data['token_type_ids'] = all_token_type_ids
    return Dataset.from_dict(data)


def _decode_ner_labels(label_ids, pred_ids, id2label):
    """Align predicted token-level IDs to word-level label sequences using the -100 mask."""
    y_true, y_pred = [], []
    for true_row, pred_row in zip(label_ids, pred_ids):
        mask = np.array(true_row) != -100
        true_seq = [id2label[int(t)] for t in np.array(true_row)[mask]]
        pred_seq = [id2label.get(int(p), 'O') for p in np.array(pred_row)[mask]]
        if true_seq:
            y_true.append(true_seq)
            y_pred.append(pred_seq)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_binary_result(preds, truth):
    tp = int(((preds == 1) & (truth == 1)).sum())
    tn = int(((preds == 0) & (truth == 0)).sum())
    fp = int(((preds == 1) & (truth == 0)).sum())
    fn = int(((preds == 0) & (truth == 1)).sum())
    result = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    result["acc"] = float((tp + tn) / (tp + tn + fp + fn))
    result["prec"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    result["rec"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    denom = result["prec"] + result["rec"]
    result["f1"] = float(2 * result["prec"] * result["rec"] / denom) if denom > 0 else 0.0
    return result


def _compute_multiclass_result(truth, preds):
    return {
        "acc": metrics.accuracy_score(truth, preds),
        "prec_micro": metrics.precision_score(truth, preds, average='micro'),
        "prec_macro": metrics.precision_score(truth, preds, average='macro'),
        "rec_micro": metrics.recall_score(truth, preds, average='micro'),
        "rec_macro": metrics.recall_score(truth, preds, average='macro'),
        "f1_micro": metrics.f1_score(truth, preds, average='micro'),
        "f1_macro": metrics.f1_score(truth, preds, average='macro'),
    }


def _compute_ner_result(y_true, y_pred):
    return {
        'f1_micro': float(f1_score(y_true, y_pred, average='micro')),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        'acc': float(accuracy_score(y_true, y_pred)),
        'prec_micro': float(precision_score(y_true, y_pred, average='micro')),
        'prec_macro': float(precision_score(y_true, y_pred, average='macro')),
        'rec_micro': float(recall_score(y_true, y_pred, average='micro')),
        'rec_macro': float(recall_score(y_true, y_pred, average='macro')),
    }


def _compute_multilabel_result(y_true, y_pred):
    return {
        "acc": example_based_accuracy(y_true, y_pred),
        "prec": example_based_precision(y_true, y_pred),
        "rec": example_based_recall(y_true, y_pred),
        "f1": example_based_f1(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Training argument builder
# ---------------------------------------------------------------------------

def _build_training_args(args, seed, save_per_epoch):
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs_per_seed,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.train_batch_size,
        fp16=False,
        seed=seed,
        eval_strategy="epoch",
        save_strategy="epoch" if save_per_epoch else "no",
        load_best_model_at_end=save_per_epoch,
        metric_for_best_model="f1" if save_per_epoch else None,
        greater_is_better=save_per_epoch,
        dataloader_num_workers=0,
        logging_strategy="epoch",
    )


def _find_checkpoint_for_epoch(output_dir, epoch):
    """Return the HF checkpoint directory that corresponds to the given epoch."""
    for name in os.listdir(output_dir):
        ckpt_dir = os.path.join(output_dir, name)
        if not os.path.isdir(ckpt_dir):
            continue
        state_file = os.path.join(ckpt_dir, 'trainer_state.json')
        if not os.path.isfile(state_file):
            continue
        with open(state_file) as f:
            state = json.load(f)
        if round(state.get('epoch', -1)) == epoch:
            return ckpt_dir
    return None


# ---------------------------------------------------------------------------
# report_per_epoch
# ---------------------------------------------------------------------------

def report_per_epoch(args, test_df, seed, model_configs):
    list_of_results = []
    pretrained = model_configs["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    # Build test dataset once outside the epoch loop
    if args.task == "multilabel":
        test_dataset = _tokenize_classification(test_df, tokenizer, args.max_seq_length, multilabel=True)
        y_true = [[int(v) for v in l] for l in test_df['labels']]
    elif args.task == "ner":
        label2id = {l: i for i, l in enumerate(args.labels_list)}
        id2label = {i: l for l, i in label2id.items()}
        test_dataset = _tokenize_ner(test_df, tokenizer, args.max_seq_length, label2id)
    else:
        test_dataset = _tokenize_classification(test_df, tokenizer, args.max_seq_length)

    for epoch in range(1, args.epochs_per_seed + 1):
        checkpoint_dir = _find_checkpoint_for_epoch(args.output_dir, epoch)
        if checkpoint_dir is None:
            print(f"\nCheckpoint not found for epoch {epoch}")
            continue

        if args.task == "multilabel":
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
            out = Trainer(model=model).predict(test_dataset)
            y_pred = (out.predictions >= 0).astype(int).tolist()
            result = _compute_multilabel_result(y_true, y_pred)

        elif args.task == "ner":
            model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
            out = Trainer(model=model).predict(test_dataset)
            y_true_ner, y_pred = _decode_ner_labels(
                np.array(test_dataset['labels']), np.argmax(out.predictions, axis=-1), id2label
            )
            result = _compute_ner_result(y_true_ner, y_pred)

        elif args.task == "binary":
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
            out = Trainer(model=model).predict(test_dataset)
            preds = np.argmax(out.predictions, axis=-1)
            result = _compute_binary_result(preds, np.array(test_df['labels']))

        elif args.task == "multiclass":
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
            out = Trainer(model=model).predict(test_dataset)
            result = _compute_multiclass_result(list(test_df['labels']), np.argmax(out.predictions, axis=-1).tolist())

        result.update({"data_name": args.dataset, "model_name": model_configs["model_name"],
                       "seed": seed, "train_batch_size": args.train_batch_size, "epoch": epoch})
        list_of_results.append(result)

    results_df = pd.DataFrame.from_dict(list_of_results, orient='columns')
    outfile_report = os.path.join("./logs/", str(args.dataset) + "_full_report.csv")
    results_df.to_csv(outfile_report, mode='a', header=not os.path.isfile(outfile_report), index=False)


# ---------------------------------------------------------------------------
# train_multi_seed
# ---------------------------------------------------------------------------

def train_multi_seed(args, train_df, eval_df, test_df, model_configs):
    init_seed = args.initial_seed
    for curr_seed in range(init_seed, init_seed + args.num_of_seeds):

        if args.task == "multilabel":
            result = train_multilabel(args, train_df, eval_df, test_df, curr_seed, model_configs)
        elif args.task == "multiclass":
            result = train_multiclass(args, train_df, eval_df, test_df, curr_seed, model_configs)
        elif args.task == "ner":
            result = train_ner(args, train_df, eval_df, test_df, curr_seed, model_configs)
        elif args.task == "binary":
            result = train_binary(args, train_df, eval_df, test_df, curr_seed, model_configs)

        log_filename = os.path.join("./logs/", args.dataset + "_best_results.json")
        out_dict = {"data_name": args.dataset, "model_name": model_configs["model_name"], "seed": curr_seed,
                    "train_batch_size": args.train_batch_size, "epochs": args.epochs_per_seed, "result": result}
        with open(log_filename, 'a') as fout:
            json.dump(out_dict, fout)
            fout.write('\n')

        if args.report_per_epoch:
            report_per_epoch(args, test_df, curr_seed, model_configs)


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_binary(args, train_df, eval_df, test_df, seed, model_configs):
    set_seed(seed)
    pretrained = model_configs["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    train_dataset = _tokenize_classification(train_df, tokenizer, args.max_seq_length)
    eval_dataset = _tokenize_classification(eval_df, tokenizer, args.max_seq_length)
    test_dataset = _tokenize_classification(test_df, tokenizer, args.max_seq_length)

    model = AutoModelForSequenceClassification.from_pretrained(pretrained, num_labels=2)

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        return {'f1': metrics.f1_score(eval_pred.label_ids, preds, average='binary', zero_division=0)}

    trainer = Trainer(model=model, args=_build_training_args(args, seed, args.report_per_epoch),
                      train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics)
    trainer.train()
    out = trainer.predict(test_dataset)
    return _compute_binary_result(np.argmax(out.predictions, axis=-1), np.array(test_df['labels']))


def train_multilabel(args, train_df, eval_df, test_df, seed, model_configs):
    set_seed(seed)
    pretrained = model_configs["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    train_dataset = _tokenize_classification(train_df, tokenizer, args.max_seq_length, multilabel=True)
    eval_dataset = _tokenize_classification(eval_df, tokenizer, args.max_seq_length, multilabel=True)
    test_dataset = _tokenize_classification(test_df, tokenizer, args.max_seq_length, multilabel=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained, num_labels=args.num_labels, problem_type="multi_label_classification"
    )

    def compute_metrics(eval_pred):
        preds = (eval_pred.predictions >= 0).astype(int)
        labels = np.array(eval_pred.label_ids).astype(int)
        return {'f1': metrics.f1_score(labels, preds, average='macro', zero_division=0)}

    trainer = Trainer(model=model, args=_build_training_args(args, seed, args.report_per_epoch),
                      train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics)
    trainer.train()

    out = trainer.predict(test_dataset)
    y_pred = (out.predictions >= 0).astype(int).tolist()
    y_true = [[int(v) for v in l] for l in test_df['labels']]

    result = _compute_multilabel_result(y_true, y_pred)
    result.update({
        "acc_labelBased": metrics.accuracy_score(y_true, y_pred),
        "prec_micro_labelBased": metrics.precision_score(y_true, y_pred, average='micro'),
        "prec_macro_labelBased": metrics.precision_score(y_true, y_pred, average='macro'),
        "rec_micro_labelBased": metrics.recall_score(y_true, y_pred, average='micro'),
        "rec_macro_labelBased": metrics.recall_score(y_true, y_pred, average='macro'),
        "f1_micro_labelBased": metrics.f1_score(y_true, y_pred, average='micro'),
        "f1_macro_labelBased": metrics.f1_score(y_true, y_pred, average='macro'),
    })
    return result


def train_multiclass(args, train_df, eval_df, test_df, seed, model_configs):
    set_seed(seed)
    pretrained = model_configs["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    train_dataset = _tokenize_classification(train_df, tokenizer, args.max_seq_length)
    eval_dataset = _tokenize_classification(eval_df, tokenizer, args.max_seq_length)
    test_dataset = _tokenize_classification(test_df, tokenizer, args.max_seq_length)

    model = AutoModelForSequenceClassification.from_pretrained(pretrained, num_labels=args.num_labels)

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        return {'f1': metrics.f1_score(eval_pred.label_ids, preds, average='macro', zero_division=0)}

    trainer = Trainer(model=model, args=_build_training_args(args, seed, args.report_per_epoch),
                      train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics)
    trainer.train()

    out = trainer.predict(test_dataset)
    return _compute_multiclass_result(list(test_df['labels']), np.argmax(out.predictions, axis=-1).tolist())


def train_ner(args, train_df, eval_df, test_df, seed, model_configs):
    set_seed(seed)
    pretrained = model_configs["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    label2id = {l: i for i, l in enumerate(args.labels_list)}
    id2label = {i: l for l, i in label2id.items()}

    train_dataset = _tokenize_ner(train_df, tokenizer, args.max_seq_length, label2id)
    eval_dataset = _tokenize_ner(eval_df, tokenizer, args.max_seq_length, label2id)
    test_dataset = _tokenize_ner(test_df, tokenizer, args.max_seq_length, label2id)

    model = AutoModelForTokenClassification.from_pretrained(
        pretrained, num_labels=len(args.labels_list), id2label=id2label, label2id=label2id
    )

    def compute_metrics(eval_pred):
        y_true, y_pred = _decode_ner_labels(
            eval_pred.label_ids, np.argmax(eval_pred.predictions, axis=-1), id2label
        )
        return {'f1': float(f1_score(y_true, y_pred, average='micro'))}

    trainer = Trainer(model=model, args=_build_training_args(args, seed, args.report_per_epoch),
                      train_dataset=train_dataset, eval_dataset=eval_dataset,
                      data_collator=DataCollatorForTokenClassification(tokenizer),
                      compute_metrics=compute_metrics)
    trainer.train()

    out = trainer.predict(test_dataset)
    y_true, y_pred = _decode_ner_labels(
        np.array(test_dataset['labels']), np.argmax(out.predictions, axis=-1), id2label
    )

    result = _compute_ner_result(y_true, y_pred)
    result['classification_report'] = str(classification_report(y_true, y_pred))
    return result


# ---------------------------------------------------------------------------
# loadData
# ---------------------------------------------------------------------------

def loadData(args):

    if args.task == "multilabel":
        train_data, eval_data, test_data = [], [], []
        num_labels = None
        for split, data in [("train.tsv", train_data), ("dev.tsv", eval_data), ("test.tsv", test_data)]:
            with open(os.path.join(args.data_dir, split)) as fd:
                for row in csv.reader(fd, delimiter="\t", quotechar='"'):
                    labels = [int(i) for i in row[1:]]
                    data.append([row[0], labels])
                    num_labels = len(labels)
        return (pd.DataFrame(train_data, columns=['text', 'labels']),
                pd.DataFrame(eval_data, columns=['text', 'labels']),
                pd.DataFrame(test_data, columns=['text', 'labels']), num_labels)

    elif args.task == "multiclass":
        train_data, eval_data, test_data = [], [], []
        max_label = 0
        for split, data in [("train.tsv", train_data), ("dev.tsv", eval_data), ("test.tsv", test_data)]:
            with open(os.path.join(args.data_dir, split)) as fd:
                for row in csv.reader(fd, delimiter="\t", quotechar='"'):
                    label = int(row[1])
                    data.append([row[0], label])
                    max_label = max(max_label, label)
        return (pd.DataFrame(train_data, columns=['text', 'labels']),
                pd.DataFrame(eval_data, columns=['text', 'labels']),
                pd.DataFrame(test_data, columns=['text', 'labels']), max_label + 1)

    elif args.task == "binary":
        train_data, eval_data, test_data = [], [], []
        for split, data in [("train.tsv", train_data), ("dev.tsv", eval_data), ("test.tsv", test_data)]:
            with open(os.path.join(args.data_dir, split)) as fd:
                for row in csv.reader(fd, delimiter="\t", quotechar='"'):
                    data.append([row[0], int(row[1])])
        return (pd.DataFrame(train_data, columns=['text', 'labels']),
                pd.DataFrame(eval_data, columns=['text', 'labels']),
                pd.DataFrame(test_data, columns=['text', 'labels']), 1)

    elif args.task == "ner":
        def _load_ner_file(path):
            data, seq = [], 0
            with open(path) as f:
                for line in f:
                    if len(line) == 1:
                        seq += 1
                    else:
                        data.append([seq] + [t.replace("\n", "") for t in line.split("\t")])
            return pd.DataFrame(data, columns=["sentence_id", "words", "labels"])

        return (_load_ner_file(os.path.join(args.data_dir, "train.txt")),
                _load_ner_file(os.path.join(args.data_dir, "dev.txt")),
                _load_ner_file(os.path.join(args.data_dir, "test.txt")), -1)

    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="insightCrime", type=str, help="The input dataset.")
    parser.add_argument("--report_per_epoch", default=False, action='store_true',
                        help="If true, will output the report per epoch.")
    args = parser.parse_args()

    with open(os.path.join("./configs/", args.dataset + ".json"), 'r') as fp:
        for k, v in json.load(fp).items():
            setattr(args, k, v)

    args.data_dir = os.path.join("./data/", args.dataset, "")
    train_df, eval_df, test_df, args.num_labels = loadData(args)

    if args.task == "ner":
        with open(os.path.join(args.data_dir, "labels.json")) as json_file:
            args.labels_list = json.load(json_file)

    for model_configs in args.models:
        args.output_dir = os.path.join("./outputs/", args.dataset, "")
        train_multi_seed(args, train_df, eval_df, test_df, model_configs)


if __name__ == "__main__":
    main()
