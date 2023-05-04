# Better Language Models of Code Through Self Improvement
The official repository for paper: [Better Language Models of Code Through Self Improvement](https://arxiv.org/pdf/2304.01228.pdf)

Findings of The 61st Annual Meeting of the Association for Computaional Linguistics (ACL 2023)

## Overview
In this paper, we propose a simple augmentation technique on input sequences from the training set to improve performance on sequence generation of code language models. In particular, after fine-tuning a pre-trained model on a specific sequence generation task, we
- Use this fine-tuned model to create an augmented version of training data and then
- Continue to fine-tune this model on the augmented dataset.
The overall training pipeline as well as process of data augmentation are depicted by the below figures.
![Overall training pipeline](assets/pipeline.pdf)
![Demonstrating the process of generating augmented datasets in our work](assets/augmentation.pdf)
## Results
Please refer to our [paper](https://arxiv.org/pdf/2304.01228.pdf) for detailed results. In summary, our experiments showed that this method, when applied to popular pre-trained code models (CodeBERT, CodeT5, and UniXCoder), significantly improves performance on code summarization and code generation tasks.
## Installation
```bash
conda create -n code-self-improve -y
conda install pip -y
pip install -r requirements.txt
```
### Downloading datasets
Our method was demonstrated on code-summarization and code-generation data from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE) benchmark. To download data dependencies, run
```bash
cd data/
./download.sh
```
## Usages
We give example script for reproducing our paper reported results. We take UniXCoder on code summarization as our demonstration.

First, navigate to the `UniXCoder` directory by

```bash
cd UniXCoder/code-summarization
```

### Example on generating augmented data
```bash
python generate_augmented_data.py \
    --model_name_or_path microsoft/unixcoder-base \
    --train_filename ./data/codesearchnet/python/train.jsonl \
    --output_dir ./augmeted_data \
    --max_soucre_length 256 \
    --max_target_length 128 \
    --beam_size 10 \
    --eval_batch_size 64
```
The augmented data will be saved into `./augmented_data/python/pseudo_data.jsonl` by running this script

### Example on fine-tuning self-improvement model
Passing the path to the augmented data derived by the above script, i.e `./augmented_data/python/pseudo_data.jsonl` to the flag `--train_filename`
```bash
python run.py \
    --do_train \
    --do_eval \
    --do_test \
    --model_name_or_path microsoft/unixcoder-base \
    --train_filename ./augmented_data/python/pseudo_data.jsonl \
    --dev_filename ./data/codesearchnet/python/valid.jsonl \
    --test_filename ./data/codesearcnet/python/test.jsonl \
    --output_dir ./output_dir \
    --max_source_length 256 \
    --max_target_length 128 \
    --beam_size 10 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 5e-6 \
    --gradient_accumulation_step 1 \
    --num_train_epoch 10
```
Final checkpoint will be saved as `./output_dir/python/checkpoint-best-bleu/pytorch_model.bin`

## Acknowledgement
Our code inherits from the following repositories:
- [CodeBERT](https://github.com/microsoft/CodeBERT)
- [UniXCoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder)
- [CodeT5](https://github.com/salesforce/CodeT5)
We thank all the researchers who have made their code publicly available to facilitate the research community as well as our work.

## Citation
```
@misc{to2023better,
      title={Better Language Models of Code through Self-Improvement}, 
      author={Hung Quoc To and Nghi D. Q. Bui and Jin Guo and Tien N. Nguyen},
      year={2023},
      eprint={2304.01228},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
