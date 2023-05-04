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

from __future__ import absolute_import
import os
import sys
from bleu import _bleu
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from CodeBLEU import calc_code_bleu
from utils import read_examples, convert_examples_to_features
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
              RobertaConfig, RobertaModel, RobertaTokenizer)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")   
  
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.") 
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--load_model_path', type=str)
    
    # print arguments
    args = parser.parse_args()
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)
    
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    # import！！！you must set is_decoder as True for generation
    config.is_decoder = True
    encoder = RobertaModel.from_pretrained(args.model_name_or_path,config=config) 

    model = Seq2Seq(encoder=encoder,decoder=encoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],eos_id=tokenizer.sep_token_id)
    
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)   
    
    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.load_model_path is not None:
        print("Loading model from {}".format(args.load_model_path))
        assert os.path.isfile(args.load_model_path)
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(args.load_model_path))                

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long) 
        train_data = TensorDataset(all_source_ids,all_target_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size // args.gradient_accumulation_steps)


        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=int(len(train_dataloader)*args.num_train_epochs*0.1),
                                                    num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        

        model.train()
        # patience, best_score, losses, dev_dataset = 0, 0, [], {}
        patience, best_score, losses, dev_dataset = 0, {"bleu": 0, "codebleu": 0}, [], {}
        for epoch in range(args.num_train_epochs):
            for idx,batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                source_ids,target_ids = batch
                loss,_,_ = model(source_ids=source_ids,target_ids=target_ids)

                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    
                losses.append(loss.item())
                loss.backward()
                if len(losses) % args.gradient_accumulation_steps == 0:
                    #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    if len(losses) // args.gradient_accumulation_steps % 100 == 0:
                        logger.info("epoch {} step {} loss {}".format(epoch,
                                                     len(losses)//args.gradient_accumulation_steps,
                                                     round(np.mean(losses[-100*args.gradient_accumulation_steps:]),4)))
            if args.do_eval:
                #Eval model with dev dataset                   
                if 'dev_loss' in dev_dataset:
                    eval_examples,eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)   
                    eval_data = TensorDataset(all_source_ids,all_target_ids)   
                    dev_dataset['dev_loss' ]= eval_examples,eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                #Start Evaling model
                model.eval()
                eval_loss,tokens_num = 0,0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,target_ids = batch                  

                    with torch.no_grad():
                        _,loss,num = model(source_ids=source_ids,target_ids=target_ids)     
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                #Pring loss of dev dataset    
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss),5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)   

                #Calculate bleu  
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long) 
                    eval_data = TensorDataset(all_source_ids)   
                    dev_dataset['dev_bleu'] = eval_examples,eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval() 
                p=[]
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids = batch[0]                  
                    with torch.no_grad():
                        preds = model(source_ids) 
                        # convert ids to text
                        for pred in preds:
                            t = pred[0].cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[:t.index(0)]
                            text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                predictions = []
                EM = []
                dev_score = dict()

                gold_fn = args.output_dir+"/dev.gold"
                output_fn = args.output_dir+"/dev.output"

                with open(output_fn,'w') as f, open(gold_fn,'w') as f1:
                    for ref,gold in zip(p,eval_examples):
                        predictions.append(ref)
                        f.write(ref+'\n')
                        f1.write(gold.target+'\n')     
                        EM.append(ref.split()==gold.target.split())   
                        
                dev_bleu = _bleu(gold_fn, output_fn) 
                dev_codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, "java")
                logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                logger.info("  %s = %s "%("EM",str(round(np.mean(EM)*100,2))))
                logger.info("  %s = %s "%("CodeBLEU",str(round(dev_codebleu*100, 4))))
                logger.info("  "+"*"*20)    

                dev_score["bleu"] = dev_bleu+round(np.mean(EM)*100,2)
                dev_score["codebleu"] = round(dev_codebleu*100,2)+round(np.mean(EM)*100,2)

                for metrics in ["bleu", "codebleu"]:
                    if dev_score[metrics]>best_score[metrics]:
                        logger.info("  Best %s score:%s", metrics, dev_score[metrics])
                        logger.info("  "+"*"*20)
                        best_score[metrics]=dev_score[metrics]
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-{}'.format(metrics))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        patience =0
                # if dev_score["bleu"]>best_score["bleu"]:
                #     logger.info("  Best bleu score:%s",dev_score["bleu"])
                #     logger.info("  "+"*"*20)
                #     best_score["bleu"]=dev_score["bleu"]
                #     # Save best checkpoint for best bleu
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                #     output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                #     torch.save(model_to_save.state_dict(), output_model_file)
                #     patience =0

                # if dev_score["codebleu"]>best_score["codebleu"]:
                #     logger.info("  Best codebleu score:%s",dev_score["codebleu"])
                #     logger.info("  "+"*"*20)
                #     best_score["codebleu"]=dev_score["codebleu"]
                #     # Save best checkpoint for best bleu
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-best-codebleu')
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                #     output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                #     torch.save(model_to_save.state_dict(), output_model_file)
                #     patience =0
                # else:
                #     patience +=1
                #     if patience == -1:
                #         break
    if args.do_test:
        logger.info("Evaluating on test dataset")
        eval_examples = read_examples(args.test_filename)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_source_ids)   

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        if args.do_train:
            list_of_metrics = ["bleu", "codebleu"]
        else:
            list_of_metrics = [None]

        for metrics in list_of_metrics:
            if args.do_train:
                logger.info("Evaluating by the best {} checkpoint".format(metrics))
                checkpoint_prefix = 'checkpoint-best-{}/pytorch_model.bin'.format(metrics)
                output_dir = os.path.join(args.output_dir, checkpoint_prefix)  
                model_to_load = model.module if hasattr(model, 'module') else model  
                model_to_load.load_state_dict(torch.load(output_dir))          

            model.eval() 
            p=[]
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids = batch[0]                  
                with torch.no_grad():
                    preds = model(source_ids)   
                    # convert ids to text
                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]
                        text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                        p.append(text)

            predictions = []
            EM = []

            # Calculate bleu
            gold_fn = args.output_dir+"/test.gold"
            output_fn = args.output_dir+"/test.output"

            with open(output_fn,'w') as f, open(gold_fn,'w') as f1:
                for ref,gold in zip(p,eval_examples):
                    predictions.append(ref)
                    f.write(ref+'\n')
                    f1.write(gold.target+'\n')     
                    EM.append(ref.split()==gold.target.split())   
                    
            test_bleu = _bleu(gold_fn, output_fn) 
            test_codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, "java")
            logger.info("  %s = %s "%("bleu-4",str(test_bleu)))
            logger.info("  %s = %s "%("EM",str(round(np.mean(EM)*100,2))))
            logger.info("  %s = %s "%("CodeBLEU",str(round(test_codebleu*100, 4))))
            logger.info("  "+"*"*20 )
            # dev_score = test_bleu+round(np.mean(EM)*100,2)

            with open(args.output_dir+"/predictions.txt",'w') as f:
                for ref,gold in zip(p,eval_examples):
                    predictions.append(str(gold.idx)+'\t'+ref)
                    f.write(ref+'\n') 

if __name__ == "__main__":
    main()
