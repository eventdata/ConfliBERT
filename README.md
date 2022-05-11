# ConfliBERT

This repository contains the essential code for the paper [ConfliBERT: A Pre-trained Language Model for Political Conflict and
Violence (NAACL 2022)](https://github.com/eventdata/ConfliBERT/blob/main/NAACL_2022___confliBERT.pdf).

## Prerequisites
The code is written by Python 3.6 in Linux system. The cuda version is 10.2. 
The necessary packages include:

	torch==1.7.1 
	transformers==4.17.0 
	numpy==1.19.2 
	scikit-learn==0.24.2
	pandas==1.1.5
	simpletransformers

## ConfliBERT Checkpoints
We provided four versions of ConfliBERT:
<ol>
  <li>ConfliBERT-scr-uncased: &nbsp;&nbsp;&nbsp;&nbsp; Pretraining from scratch with our own uncased vocabulary (preferred)</li>
  <li>ConfliBERT-scr-cased: &nbsp;&nbsp;&nbsp;&nbsp; Pretraining from scratch with our own cased vocabulary</li>
  <li>ConfliBERT-cont-uncased: &nbsp;&nbsp;&nbsp;&nbsp; Continual pretraining with original BERT's uncased vocabulary</li>
  <li>ConfliBERT-cont-cased: &nbsp;&nbsp;&nbsp;&nbsp; Continual pretraining with original BERT's cased vocabulary</li>
</ol>


You can import the above four models directly via Huggingface API:

	from transformers import AutoTokenizer, AutoModelForMaskedLM
	tokenizer = AutoTokenizer.from_pretrained("snowood1/ConfliBERT-scr-uncased", use_auth_token=True)
	model = AutoModelForMaskedLM.from_pretrained("snowood1/ConfliBERT-scr-uncased", use_auth_token=True)


## Evaluation	
The usage of ConfliBERT is the same as other BERT models in Huggingface.

We provided multiple examples using [Simple Transformers](https://simpletransformers.ai/).

The 1st step is to preprocess the datasets into the required formats in [./data](https://github.com/eventdata/ConfliBERT/tree/main/data). For example,

<ol>
  <li>IndiaPoliceEvents_sents for classfication tasks. The format is sentence + labels separated by tabs.</li>
  <li>re3d for NER tasks in CONLL format</li>
</ol>

The 2nd step is to create the corresponding config files in [./configs](https://github.com/eventdata/ConfliBERT/tree/main/configs) with the correct tasks from ["binary", "multiclass", "multilabel", "ner"].

Finally, you can run
	
	CUDA_VISIBLE_DEVICES=0 python finetune_data.py --dataset IndiaPoliceEvents_sents --report_per_epoch

	
## Pretraining Corpus
We have gathered a large corpus in politics and conflicts domain (33 GB) for pretraining ConfliBERT.
The folder [./pretrain-corpora/Crawlers and Processes](https://github.com/eventdata/ConfliBERT/tree/main/pretrain-corpora/Crawlers%20and%20Process) contains the sample scripts used to generate the corpus used in this study. 
Due to the copyright, we provide a few samples in [./pretrain-corpora/Samples](https://github.com/eventdata/ConfliBERT/tree/main/pretrain-corpora/Samples).  These samples follow the format of "one sentence per line format". See more details of pretraining corpora in our paper's Section 2 and Appendix.




## Pretraining Scripts
We followed the same pretraining scripts run_mlm.py from Huggingface [(The original link)](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py).
Below is an example using 8 GPUs. We have provided our parameters in the Appendix. However, you should change the parameters according to your own devices:
<details>
	
```	
	export NGPU=8; nohup python -m torch.distributed.launch --master_port 12345 \
	--nproc_per_node=$NGPU run_mlm.py \
	--model_type bert \
	--config_name ./bert_base_cased \
	--tokenizer_name ./bert_base_cased \
	--output_dir ./bert_base_cased \
	--cache_dir ./cache_cased_128 \
	--use_fast_tokenizer \
	--overwrite_output_dir \
	--train_file YOUR_TRAIN_FILE \
	--validation_file YOUR_VALID_FILE \
	--max_seq_length 128\ 
	--preprocessing_num_workers 4 \
	--dataloader_num_workers 2 \
	--do_train --do_eval \
	--learning_rate 5e-4 \
	--warmup_steps=10000 \
	--save_steps 1000 \
	--evaluation_strategy steps \
	--eval_steps 10000 \
	--prediction_loss_only  \
	--save_total_limit 3 \
	--per_device_train_batch_size 64 --per_device_eval_batch_size 64 \
	--gradient_accumulation_steps 4 \
	--logging_steps=100 \
	--max_steps 100000 \
	--adam_beta1 0.9 --adam_beta2 0.98 --adam_epsilon 1e-6 \
	--fp16 True --weight_decay=0.01
```
	
</details>

## Citation

If you find this repo useful in your research, please consider citing:

	@inproceedings{conflibert2022,
	  title={ConfliBERT: A Pre-trained Language Model for Political Conflict and Violence},
	  author={Hu, Yibo and Hosseini, Mohammad Saleh and Skorupa Parolin, Erick and Osorio, Javier and Khan, Latifur and Brandt, Patrick and D'Orazio, Vito},
	  booktitle={Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
	  year={2022}
	}
