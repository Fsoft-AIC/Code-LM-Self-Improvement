# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
import json

import torch
from torch.utils.data import DataLoader, SequentialSampler
from models import build_or_load_gen_model
from evaluator import smooth_bleu
# from evaluator.CodeBLEU import calc_code_bleu
# from evaluator.bleu import _bleu
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    logger.info("  " + "***** Testing *****")
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_examples, eval_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'test',
                                                       only_src=True, is_sample=False)
    print("len eval_data: {}".format(len(eval_data)))
    logger.info("  ***** Running bleu evaluation on train data*****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pseudo_data = []
    k_best_lists = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for train set"):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            preds = model.generate(source_ids,
                                   attention_mask=source_mask,
                                   use_cache=True,
                                   num_beams=args.beam_size,
                                   early_stopping=args.task == 'summarize',
                                   max_length=args.max_target_length,
                                   num_return_sequences=args.beam_size,
                                   )
            preds = preds.cpu().tolist()
            # batching the predictions
            preds = [preds[i:i+args.beam_size] for i in range(0, len(preds), args.beam_size)]
            _k_best_lists = [[tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for ids in batch_ids] for batch_ids in preds]
            k_best_lists.extend(_k_best_lists)

    # Calculating similarity
    for k_best_list, ex in zip(k_best_lists, eval_examples):
        scores = [smooth_bleu.bleu([ex.target], pred)[0] for pred in k_best_list] 
        index = np.argmax(scores)
        y_hat = k_best_list[index]
        score = scores[index]
        pseudo_data.append((ex.source, y_hat, score))

    # output_dir = os.path.join(args.output_dir, args.sub_task)
    objects = [{"code_tokens": code.strip().split(),
                "docstring_tokens": nl.strip().split(),
               "sim_score": sim_score}
               for code, nl, sim_score in pseudo_data]
    output_filepath = os.path.join(args.output_dir, "pseudo_data.jsonl")
    with open(output_filepath, "w") as f:
        for obj in objects:
            json.dump(obj, f)
            f.write("\n")

    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()

if __name__ == "__main__":
    main()
