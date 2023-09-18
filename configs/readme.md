This folder contains configuration files for running experiments. 
For example, when conducting experiments on the 'IndiaPoliceEvents_sents' dataset, you should use the 'IndiaPoliceEvents_sents.json' configuration file.


### Example Files

- **IndiaPoliceEvents_sents.json:** An example file for classification tasks.

- **re3d.json:** An example file for Named Entity Recognition (NER) tasks.

## Configuration Example (IndiaPoliceEvents_sents.json)

```json
{
    "task": "multilabel",
    "num_of_seeds": 3,
    "initial_seed": 123,
    "epochs_per_seed": 5,
    "train_batch_size": 16,
    "max_seq_length": 128,
    "models": [
        {
            "model_name": "ConfliBERT-scr-uncased",
            "model_path": "snowood1/ConfliBERT-scr-uncased",
            "architecture": "bert",
            "do_lower_case": true
        },
        {
            "model_name": "bert-base-cased",
            "model_path": "bert-base-cased",
            "architecture": "bert",
            "do_lower_case": false
        }
    ]
}
```
### Experiment Options

- **"tasks"**: Choose from the following task types: ["binary", "multiclass", "multilabel", "ner"] for the dataset. Below are the defined task for the processed datasets in the paper:

| Dataset | Task |
| :-------- | :-------- |
|20news| binary |
| BBC_News | binary |
| IndiaPoliceEvents_doc | multilabel |
| IndiaPoliceEvents_sents | multilabel|
| cameo_class | multiclass |
| cameo_ner | ner |
| insightCrime | multilabel |
| re3d | ner |
| satp_relevant | multilabel|


- **"num_of_seeds"**: Decide how many experiments you want to repeat with different random seeds to calculate average results for analysis.

- **"initial_seed"**: Specify the initial random seed number.

- **"epochs_per_seed"**: Define the number of epochs for each experiment with a random seed during training.

- **"train_batch_size"**: Set the training batch size.

- **"max_seq_length"**: Determine the maximum sequence length.
  
- **"models"**: Model selection.

### Model Selection

You have the flexibility to choose from a list of available models for your experiments. If you wish to experiment with just one model, simply include that single model. If you want to experiment with multiple models, you can add them to the list.

For example, our example configuration includes two models: "ConfliBERT-scr-uncased" and "bert-cased." Both of these models will be run with the same set of hyperparameters.

The **model_path** refers to its Hugging Face model card address, such as [https://huggingface.co/snowood1/ConfliBERT-scr-uncased](https://huggingface.co/snowood1/ConfliBERT-scr-uncased).

We have tested six models in our experiments:

```json

        {
            "model_name": "ConfliBERT-scr-cased",
            "model_path": "snowood1/ConfliBERT-scr-cased",
            "architecture": "bert",
            "do_lower_case": false
        },
        {
            "model_name": "ConfliBERT-scr-uncased",
            "model_path": "snowood1/ConfliBERT-scr-uncased",
            "architecture": "bert",
            "do_lower_case": true
        },
        {
            "model_name": "ConfliBERT-cont-cased",
            "model_path": "snowood1/ConfliBERT-cont-cased",
            "architecture": "bert",
            "do_lower_case": false
        },
        {
            "model_name": "ConfliBERT-cont-uncased",
            "model_path": "snowood1/ConfliBERT-cont-uncased",
            "architecture": "bert",
            "do_lower_case": true
        },
        {
            "model_name": "bert-base-cased",
            "model_path": "bert-base-cased",
            "architecture": "bert",
            "do_lower_case": false
        },
        {
            "model_name": "bert-base-uncased",
            "model_path": "bert-base-uncased",
            "architecture": "bert",
            "do_lower_case": true
        }

```

Besides the above six models, any pretrained model of that type found in the [Hugging Face docs](https://huggingface.co/transformers/v3.3.1/pretrained_models.html) should work. The architecture type available for each task can be found under their respective section. To use any of them set the correct architecture and model_name in the args dictionary.
