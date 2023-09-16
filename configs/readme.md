This folder contains configuration files for running experiments. 
For example, when conducting experiments on the 'indiapoliceevent_sents' dataset, you should use the 'IndiaPoliceEvents_sents.json' configuration file.


### Example Files

- **IndiaPoliceEvents_sents.json:** An example file for classification tasks.

- **re3d.json:** An example file for Named Entity Recognition (NER) tasks.

## Configuration Example (IndiaPoliceEvents_sents.json)

```json
{
    "task": "multilabel",
    "num_of_seeds": 5,
    "initial_seed": 123,
    "epochs_per_seed": 5,
    "train_batch_size": 16,
    "max_seq_length": 128,
    "models": [
        {
            "model_name": "ConfliBERT-scr-cased",
            "model_path": "snowood1/ConfliBERT-scr-cased",
            "architecture": "bert",
            "do_lower_case": false
        },
        {
            "model_name": "bert-base-cased",
            "model_path": "snowood1/ConfliBERT-scr-cased",
            "architecture": "bert",
            "do_lower_case": true
        }
}
```
### Experiment Options

- **"tasks"**: Choose from the following task types: ["binary", "multiclass", "multilabel", "ner"]

- **"num_of_seeds"**: Decide how many experiments you want to repeat with different random seeds to calculate average results for analysis.

- **"initial_seed"**: Specify the initial random seed number.

- **"epochs_per_seed"**: Define the number of epochs for each experiment with a random seed during training.

- **"train_batch_size"**: Set the training batch size.

- **"max_seq_length"**: Determine the maximum sequence length.

### Model Selection

You have the flexibility to choose from a list of available models for your experiments. If you wish to experiment with just one model, simply include that single model. If you want to experiment with multiple models, you can add them to the list.

For example, our example configuration includes two models: "ConfliBERT-scr-uncased" and "bert-cased." Both of these models will be run with the same set of hyperparameters.

The **model_path** refers to its Hugging Face model card address, such as [https://huggingface.co/snowood1/ConfliBERT-scr-uncased](https://huggingface.co/snowood1/ConfliBERT-scr-uncased).

