from simpletransformers.classification import MultiLabelClassificationModel, MultiLabelClassificationArgs, ClassificationModel, ClassificationArgs
from simpletransformers.ner import NERModel, NERArgs
from example_based import example_based_accuracy, example_based_recall, example_based_precision, example_based_f1
from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import argparse
from sklearn import metrics
import pandas as pd
import numpy as np
import csv
import os
import json



def report_per_epoch(args, test_df, seed, model_configs):

    list_of_results = []
    for epoch in range(1, args.epochs_per_seed+1):
        
        end_dir = "epoch-"+str(epoch)
        
        dirs = [f for f in os.listdir(args.output_dir) if f[-len(end_dir):] == end_dir and os.path.isdir(os.path.join(args.output_dir, f))]
        
        if len(dirs) == 0:
            print("\nCheckpoint not found for epoch", str(epoch))
        
        else:
            checkpoint_dir = os.path.join(args.output_dir, dirs[0])

            if args.task == "multilabel":
                model = MultiLabelClassificationModel(model_configs["architecture"], checkpoint_dir)

                ## Performance data: Evaluating the model on test data
                predictions, raw_outputs = model.predict(test_df.text)
                result_test, model_outputs, wrong_predictions = model.eval_model(test_df)
    
                result = {k: float(v) for k, v in result_test.items()}
                result["acc"] = example_based_accuracy([list(test_df.labels[i]) for i, pred in enumerate(predictions)], [list(pred) for pred in predictions])
                result["prec"] = example_based_precision([list(test_df.labels[i]) for i, pred in enumerate(predictions)], [list(pred) for pred in predictions])
                result["rec"] = example_based_recall([list(test_df.labels[i]) for i, pred in enumerate(predictions)], [list(pred) for pred in predictions])
                result["f1"] = example_based_f1([list(test_df.labels[i]) for i, pred in enumerate(predictions)], [list(pred) for pred in predictions])
            
            elif args.task == "ner":
                
                model = NERModel(model_configs["architecture"], checkpoint_dir)

                ## Performance data: Evaluating the model on test data
                # Getting true labels
                labels = []
                temp = []
                seq = test_df['sentence_id'].tolist()[0]

                for lab, id in zip(test_df['labels'].tolist(), test_df['sentence_id'].tolist()):
                    if id != seq:
                        seq = id
                        labels.append(temp)
                        temp = []

                    temp.append(lab)
                labels.append(temp)

                # Evaluating the model on test data
                result, model_outputs, preds = model.eval_model(test_df)

                # Computing performance metrics thru seqeval
                y_pred = []
                y_true = []
                for pred, true in zip(preds, labels):
                    if len(pred) == len(true):
                        y_pred.append(pred)
                        y_true.append(true)
                
                result = {}
                result['f1_micro'] = float(f1_score(y_true, y_pred, average='micro'))
                result['f1_macro'] = float(f1_score(y_true, y_pred, average='macro'))
                result['acc'] = float(accuracy_score(y_true, y_pred))
                result['prec_micro'] = float(precision_score(y_true, y_pred, average='micro'))
                result['prec_macro'] = float(precision_score(y_true, y_pred, average='macro'))
                result['rec_micro'] = float(recall_score(y_true, y_pred, average='micro'))
                result['rec_macro'] = float(recall_score(y_true, y_pred, average='macro'))
            
            elif args.task == "binary":
                
                # Create a MultiLabelClassificationModel
                model = ClassificationModel(model_configs["architecture"], checkpoint_dir)

                # Evaluating the model on test data
                result_np, model_outputs, wrong_predictions = model.eval_model(test_df)

                # Collecting relevant results
                result = {}
                for k, v in result_np.items():
                    if k in ["mcc", "auroc", "auprc", "eval_loss"]:
                        result[k] = float(v)
                    else:
                        result[k] = int(v)

                result["acc"] = float((result["tp"]+result["tn"])/(result["tp"]+result["tn"]+result["fp"]+result["fn"]))
                result["prec"] = float((result["tp"])/(result["tp"]+result["fp"]))
                result["rec"] = float((result["tp"])/(result["tp"]+result["fn"]))
                result["f1"] = float(2*(result["prec"]*result["rec"])/(result["prec"]+result["rec"]))

        
            elif args.task == "multiclass":

                # Create a MultiLabelClassificationModel
                model = ClassificationModel(model_configs["architecture"], checkpoint_dir)

                # Evaluating the model on test data
                predictions, raw_outputs = model.predict(test_df.text)
                truth = list(test_df.labels)
                result_np, model_outputs, wrong_predictions = model.eval_model(test_df)

                # Collecting relevant results
                result = {k: float(v) for k, v in result_np.items()}

                result["acc"] = metrics.accuracy_score(truth, predictions)
                result["prec_micro"] = metrics.precision_score(truth, predictions, average='micro')
                result["prec_macro"] = metrics.precision_score(truth, predictions, average='macro')
                result["rec_micro"] = metrics.recall_score(truth, predictions, average='micro')
                result["rec_macro"] = metrics.recall_score(truth, predictions, average='macro')
                result["f1_micro"] = metrics.f1_score(truth, predictions, average='micro')
                result["f1_macro"] = metrics.f1_score(truth, predictions, average='macro')


            ## Other relevant information
            result["data_name"] = args.dataset
            result["model_name"] = model_configs["model_name"]
            result["seed"] = seed
            result["train_batch_size"] = args.train_batch_size
            result["epoch"] = epoch

            list_of_results.append(result)


    results_df = pd.DataFrame.from_dict(list_of_results, orient='columns')
    outfile_report = os.path.join("./logs/", str(args.dataset)+"_full_report.csv")

    if os.path.isfile(outfile_report):
        results_df.to_csv(outfile_report, mode='a', header=False, index=False)
    else:
        results_df.to_csv(outfile_report, mode='a', header=True, index=False)






def train_multi_seed(args, train_df, eval_df, test_df, model_configs):

    init_seed = args.initial_seed
    for curr_seed in range(init_seed, init_seed + args.num_of_seeds):
        
        if args.task == "multilabel":
            result = train_multilabel(args, train_df, eval_df, test_df, curr_seed, model_configs)
        if args.task == "multiclass":
            result = train_multiclass(args, train_df, eval_df, test_df, curr_seed, model_configs)
        elif args.task == "ner":
            result = train_ner(args, train_df, eval_df, test_df, curr_seed, model_configs)
        elif args.task == "binary":
            result = train_binary(args, train_df, eval_df, test_df, curr_seed, model_configs)

        # Recording best results for a given seed
        log_filename = os.path.join("./logs/", args.dataset+"_best_results.json")
        out_dict = {"data_name":args.dataset, "model_name": model_configs["model_name"], "seed": curr_seed, "train_batch_size": args.train_batch_size, "epochs": args.epochs_per_seed, "result": result}

        with open(log_filename, 'a') as fout:
            json.dump(out_dict, fout)
            fout.write('\n')
        
        if args.report_per_epoch:
            report_per_epoch(args, test_df, curr_seed, model_configs)



def train_binary(args, train_df, eval_df, test_df, seed, model_configs):

    # Training Arguments
    model_args = ClassificationArgs()
    model_args.manual_seed = seed
    model_args.best_model_dir = os.path.join(args.output_dir, "best_model", "")
    model_args.output_dir =  args.output_dir
    model_args.num_train_epochs = args.epochs_per_seed
    model_args.fp16 = False
    model_args.max_seq_length = args.max_seq_length
    model_args.train_batch_size = args.train_batch_size
    model_args.save_steps = -1
    model_args.use_multiprocessing = False
    # model_args.save_model_every_epoch = False
    if "do_lower_case" in model_configs:
        model_args.do_lower_case = model_configs["do_lower_case"]
    model_args.evaluate_during_training = True
    model_args.save_best_model = True
    model_args.save_eval_checkpoints = False
    # model_args.no_save = True
    model_args.overwrite_output_dir = True

    if not args.report_per_epoch:
        model_args.save_model_every_epoch = False
        model_args.no_save = True

    # Create a MultiLabelClassificationModel
    architecture = model_configs["architecture"]
    pretrained_model = model_configs["model_path"]
    model = ClassificationModel(architecture, pretrained_model, args=model_args)

    # Train the model
    model.train_model(train_df, eval_df=eval_df)

    # Evaluating the model on test data
    result_np, model_outputs, wrong_predictions = model.eval_model(test_df)
    
    # Collecting relevant results
    result = {}
    for k, v in result_np.items():
        if k in ["mcc", "auroc", "auprc", "eval_loss"]:
            result[k] = float(v)
        else:
            result[k] = int(v)
    
    result["acc"] = float((result["tp"]+result["tn"])/(result["tp"]+result["tn"]+result["fp"]+result["fn"]))
    result["prec"] = float((result["tp"])/(result["tp"]+result["fp"]))
    result["rec"] = float((result["tp"])/(result["tp"]+result["fn"]))
    result["f1"] = float(2*(result["prec"]*result["rec"])/(result["prec"]+result["rec"]))

    return result


def train_multilabel(args, train_df, eval_df, test_df, seed, model_configs):

    # Training Arguments
    model_args = MultiLabelClassificationArgs()
    model_args.manual_seed = seed
    model_args.best_model_dir = os.path.join(args.output_dir, "best_model", "")
    model_args.output_dir =  args.output_dir
    model_args.num_train_epochs = args.epochs_per_seed
    model_args.fp16 = False
    model_args.max_seq_length = args.max_seq_length
    model_args.train_batch_size = args.train_batch_size
    model_args.save_steps = -1
    model_args.use_multiprocessing = False
    # model_args.save_model_every_epoch = False
    if "do_lower_case" in model_configs:
        model_args.do_lower_case = model_configs["do_lower_case"]
    model_args.evaluate_during_training = True
    model_args.save_best_model = True
    model_args.save_eval_checkpoints = False
    # model_args.no_save = True
    model_args.overwrite_output_dir = True

    if not args.report_per_epoch:
        model_args.save_model_every_epoch = False
        model_args.no_save = True



    # Create a MultiLabelClassificationModel
    architecture = model_configs["architecture"]
    pretrained_model = model_configs["model_path"]
    model = MultiLabelClassificationModel(architecture, pretrained_model, num_labels=args.num_labels, args=model_args)

    # Train the model
    model.train_model(train_df, eval_df=eval_df) 

    # Evaluatinge the model on test data
    predictions, raw_outputs = model.predict(test_df.text)
    result_test, model_outputs, wrong_predictions = model.eval_model(test_df)

    # Collecting relevant results
    result = {k: float(v) for k, v in result_test.items()}
    y_true = [list(test_df.labels[i]) for i, pred in enumerate(predictions)]
    y_pred = [list(pred) for pred in predictions]
    result["acc"] = example_based_accuracy(y_true, y_pred)
    result["prec"] = example_based_precision(y_true, y_pred)
    result["rec"] = example_based_recall(y_true, y_pred)
    result["f1"] = example_based_f1(y_true, y_pred)

    # Computing performance through scikit metrics
    result["acc_labelBased"] = metrics.accuracy_score(y_true, y_pred)
    result["prec_micro_labelBased"] = metrics.precision_score(y_true, y_pred, average='micro')
    result["prec_macro_labelBased"] = metrics.precision_score(y_true, y_pred, average='macro')
    result["rec_micro_labelBased"] = metrics.recall_score(y_true, y_pred, average='micro')
    result["rec_macro_labelBased"] = metrics.recall_score(y_true, y_pred, average='macro')
    result["f1_micro_labelBased"] = metrics.f1_score(y_true, y_pred, average='micro')
    result["f1_macro_labelBased"] = metrics.f1_score(y_true, y_pred, average='macro')



    return result








def train_multiclass(args, train_df, eval_df, test_df, seed, model_configs):

    # Training Arguments
    model_args = ClassificationArgs()
    model_args.manual_seed = seed
    model_args.best_model_dir = os.path.join(args.output_dir, "best_model", "")
    model_args.output_dir =  args.output_dir
    model_args.num_train_epochs = args.epochs_per_seed
    model_args.fp16 = False
    model_args.max_seq_length = args.max_seq_length
    model_args.train_batch_size = args.train_batch_size
    model_args.save_steps = -1
    model_args.use_multiprocessing = False
    # model_args.save_model_every_epoch = False
    if "do_lower_case" in model_configs:
        model_args.do_lower_case = model_configs["do_lower_case"]
    model_args.evaluate_during_training = True
    model_args.save_best_model = True
    model_args.save_eval_checkpoints = False
    # model_args.no_save = True
    model_args.overwrite_output_dir = True

    if not args.report_per_epoch:
        model_args.save_model_every_epoch = False
        model_args.no_save = True

    # Create a MultiLabelClassificationModel
    architecture = model_configs["architecture"]
    pretrained_model = model_configs["model_path"]
    model = ClassificationModel(architecture, pretrained_model, num_labels=args.num_labels, args=model_args)

    # Train the model
    model.train_model(train_df, eval_df=eval_df)

    # Evaluating the model on test data
    predictions, raw_outputs = model.predict(test_df.text)
    truth = list(test_df.labels)
    result_np, model_outputs, wrong_predictions = model.eval_model(test_df)
    

    # Collecting relevant results
    result = {k: float(v) for k, v in result_np.items()}

    result["acc"] = metrics.accuracy_score(truth, predictions)
    result["prec_micro"] = metrics.precision_score(truth, predictions, average='micro')
    result["prec_macro"] = metrics.precision_score(truth, predictions, average='macro')
    result["rec_micro"] = metrics.recall_score(truth, predictions, average='micro')
    result["rec_macro"] = metrics.recall_score(truth, predictions, average='macro')
    result["f1_micro"] = metrics.f1_score(truth, predictions, average='micro')    
    result["f1_macro"] = metrics.f1_score(truth, predictions, average='macro')

    return result

















def train_ner(args, train_df, eval_df, test_df, seed, model_configs):


    # Training Arguments
    model_args = NERArgs()
    model_args.labels_list = args.labels_list
    model_args.manual_seed = seed
    model_args.best_model_dir = os.path.join(args.output_dir, "best_model", "")
    model_args.output_dir =  args.output_dir
    model_args.num_train_epochs =  args.epochs_per_seed
    model_args.fp16 = False
    model_args.max_seq_length = args.max_seq_length
    model_args.train_batch_size = args.train_batch_size
    model_args.save_steps = -1
    model_args.use_multiprocessing = False
    # model_args.save_model_every_epoch = False
    if "do_lower_case" in model_configs:
        model_args.do_lower_case = model_configs["do_lower_case"]
    model_args.evaluate_during_training = True
    model_args.save_best_model = True
    model_args.save_eval_checkpoints = False
    # model_args.no_save = True
    model_args.overwrite_output_dir = True

    if not args.report_per_epoch:
        model_args.save_model_every_epoch = False
        model_args.no_save = True



    # Create a NERModel
    architecture = model_configs["architecture"]
    pretrained_model = model_configs["model_path"]
    model = NERModel(architecture, pretrained_model, args=model_args)

    # Train the model
    model.train_model(train_df, eval_df=eval_df)

    
    # Getting true labels
    labels = []
    temp = []
    seq = test_df['sentence_id'].tolist()[0]

    for lab, id in zip(test_df['labels'].tolist(), test_df['sentence_id'].tolist()):
        if id != seq:
            seq = id
            labels.append(temp)
            temp = []

        temp.append(lab)
    labels.append(temp)

    # Evaluating the model on test data
    result, model_outputs, preds = model.eval_model(test_df)

    # Computing performance metrics thru seqeval
    y_pred = []
    y_true = []
    for pred, true in zip(preds, labels):
        if len(pred) == len(true):
            y_pred.append(pred)
            y_true.append(true)


    result_seqeval = {}
    result_seqeval['f1_micro'] = float(f1_score(y_true, y_pred, average='micro'))
    result_seqeval['f1_macro'] = float(f1_score(y_true, y_pred, average='macro'))
    result_seqeval['acc'] = float(accuracy_score(y_true, y_pred))
    result_seqeval['prec_micro'] = float(precision_score(y_true, y_pred, average='micro'))
    result_seqeval['prec_macro'] = float(precision_score(y_true, y_pred, average='macro'))
    result_seqeval['rec_micro'] = float(recall_score(y_true, y_pred, average='micro'))
    result_seqeval['rec_macro'] = float(recall_score(y_true, y_pred, average='macro'))
    result_seqeval['classification_report'] = str(classification_report(y_true, y_pred))

    return result_seqeval










def loadData(args):

    if args.task == "multilabel":

        train_data = []
        with open(os.path.join(args.data_dir, "train.tsv")) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                text = row[0]
                labels = [int(i) for i in row[1:]]
                train_data.append([text, labels])
        train_df = pd.DataFrame(train_data, columns=['text', 'labels'])
        num_labels = len(labels)

        eval_data = []
        with open(os.path.join(args.data_dir, "dev.tsv")) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                text = row[0]
                labels = [int(i) for i in row[1:]]
                eval_data.append([text, labels])
        eval_df = pd.DataFrame(eval_data, columns=['text', 'labels'])

        test_data = []
        with open(os.path.join(args.data_dir, "test.tsv")) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                text = row[0]
                labels = [int(i) for i in row[1:]]
                test_data.append([text, labels])

        test_df = pd.DataFrame(test_data, columns=['text', 'labels'])

        return train_df, eval_df, test_df, num_labels


    '''
    if args.task == "multiclass":

        train_data = []
        max_label = 0
        with open(os.path.join(args.data_dir, "train.tsv")) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                text = row[0]
                labels = [int(i) for i in row[1:]]
                max_label = max(max_label,max(labels))
                train_data.append([text, labels])
        train_df = pd.DataFrame(train_data, columns=['text', 'labels'])
        num_labels = max_label+1

        eval_data = []
        with open(os.path.join(args.data_dir, "dev.tsv")) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                text = row[0]
                labels = [int(i) for i in row[1:]]
                eval_data.append([text, labels])
        eval_df = pd.DataFrame(eval_data, columns=['text', 'labels'])

        test_data = []
        with open(os.path.join(args.data_dir, "test.tsv")) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                text = row[0]
                labels = [int(i) for i in row[1:]]
                test_data.append([text, labels])

        test_df = pd.DataFrame(test_data, columns=['text', 'labels'])

        return train_df, eval_df, test_df, num_labels
    '''
    if args.task == "multiclass":

        train_data = []
        max_label = 0
        with open(os.path.join(args.data_dir, "train.tsv")) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                text = row[0]
                labels = int(row[1:][0])
                max_label = max(max_label,labels)
                train_data.append([text, labels])
        train_df = pd.DataFrame(train_data, columns=['text', 'labels'])
        num_labels = max_label+1

        eval_data = []
        with open(os.path.join(args.data_dir, "dev.tsv")) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                text = row[0]
                labels = int(row[1:][0])
                eval_data.append([text, labels])
        eval_df = pd.DataFrame(eval_data, columns=['text', 'labels'])

        test_data = []
        with open(os.path.join(args.data_dir, "test.tsv")) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                text = row[0]
                labels = int(row[1:][0])
                test_data.append([text, labels])

        test_df = pd.DataFrame(test_data, columns=['text', 'labels'])

        return train_df, eval_df, test_df, num_labels





    elif args.task == "binary":
        train_data = []
        with open(os.path.join(args.data_dir, "train.tsv")) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                text = row[0]
                labels = int(row[1])
                train_data.append([text, labels])
        train_df = pd.DataFrame(train_data, columns=['text', 'labels'])

        eval_data = []
        with open(os.path.join(args.data_dir, "dev.tsv")) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                text = row[0]
                labels = int(row[1])
                eval_data.append([text, labels])
        eval_df = pd.DataFrame(eval_data, columns=['text', 'labels'])

        test_data = []
        with open(os.path.join(args.data_dir, "test.tsv")) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                text = row[0]
                labels = int(row[1])
                test_data.append([text, labels])

        test_df = pd.DataFrame(test_data, columns=['text', 'labels'])
 
        return train_df, eval_df, test_df, 1


    elif args.task == "ner":

        train_data = []
        seq = 0
        with open(os.path.join(args.data_dir, "train.txt")) as f:
            lines = f.readlines()
            for line in lines:
                if len(line) == 1:
                    seq = seq + 1
                else:
                    train_data.append([seq]+[l.replace("\n", "") for l in line.split("\t")])

        train_df = pd.DataFrame(train_data, columns=["sentence_id", "words", "labels"])


        dev_data = []
        seq = 0
        with open(os.path.join(args.data_dir, "dev.txt")) as f:
            lines = f.readlines()
            for line in lines:
                if len(line) == 1:
                    seq = seq + 1
                else:
                    dev_data.append([seq]+[l.replace("\n", "") for l in line.split("\t")])
    
        eval_df = pd.DataFrame(dev_data, columns=["sentence_id", "words", "labels"])

        test_data = []
        seq = 0
        with open(os.path.join(args.data_dir, "test.txt")) as f:
            lines = f.readlines()
            for line in lines:
                if len(line) == 1:
                    seq = seq + 1
                else:
                    test_data.append([seq]+[l.replace("\n", "") for l in line.split("\t")])

        test_df = pd.DataFrame(test_data, columns=["sentence_id", "words", "labels"])

        return train_df, eval_df, test_df, -1


    return None










def main():

    parser = argparse.ArgumentParser()

    ## Main parameters
    parser.add_argument("--dataset",
                        default = "insightCrime",
                        type=str,
                        help="The input dataset.")
    parser.add_argument("--report_per_epoch",
                        default = False,
                        action='store_true',
                        help="If true, will output the report per epoch.")


    args = parser.parse_args()


    ## Loading the configurations and relevant variables
    data_json = os.path.join("./configs/", args.dataset+".json")
    with open(data_json, 'r') as fp:
        data_configs = json.load(fp)

    for k,v in data_configs.items():
        setattr(args, k, v)

    args.data_dir = os.path.join("./data/", args.dataset, "")


    ## Loading the datasets
    train_df, eval_df, test_df, args.num_labels = loadData(args)
    
    if args.task == "ner":
        with open(os.path.join(args.data_dir, "labels.json")) as json_file:
            args.labels_list = json.load(json_file)

    
    ## Running experiments for all the models in configs:
    for model_configs in args.models:
        
        # args.output_dir = os.path.join("./outputs/", args.dataset + "_" + model_configs["model_name"], "")
        args.output_dir = os.path.join("./outputs/", args.dataset, "")

        train_multi_seed(args, train_df, eval_df, test_df, model_configs)
        


if __name__ == "__main__":
    main()
