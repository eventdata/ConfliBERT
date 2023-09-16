### Example Files

- **IndiaPoliceEvents_sents.json:** An example file for classification tasks.

- **re3d.json:** An example file for Named Entity Recognition (NER) tasks.

### Experiment Options

- **"tasks"**: Choose from the following task types: ["binary", "multiclass", "multilabel", "ner"]

- **"num_of_seeds"**: Decide how many experiments you want to repeat with different random seeds to calculate average results for analysis.

- **"initial_seed"**: Specify the initial random seed number.

- **"epochs_per_seed"**: Define the number of epochs for each experiment with a random seed during training.

- **"train_batch_size"**: Set the training batch size.

- **"max_seq_length"**: Determine the maximum sequence length.

### Model Selection

You can select from a list of available models. For example, "ConfliBERT-scr-uncased" is one of the options:

```json
{
    "model_name": "ConfliBERT-scr-uncased",
    "model_path": "snowood1/ConfliBERT-scr-uncased", 
    "architecture": "bert",
    "do_lower_case": true, 
}
```
Where model_path refers to its Hugging Face model card address:  https://huggingface.co/snowood1/ConfliBERT-scr-uncased.
