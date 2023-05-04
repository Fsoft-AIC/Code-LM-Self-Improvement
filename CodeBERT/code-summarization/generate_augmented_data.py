import os
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm
import json

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler#, RandomSampler
from model import Seq2Seq
from utils import read_examples, convert_examples_to_features
import smooth_bleu
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
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
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--load_model_path', type=str)
    
    # print arguments
    args = parser.parse_args()
    assert not os.path.exists(os.path.join(args.output_dir, "checkpoint-best-bleu", "pytorch_model.bin"))
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
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(args.load_model_path))                

    eval_examples = read_examples(args.train_filename)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_source_ids)   

    # generate_pseudo_data(args, eval_examples, eval_data, model, tokenizer)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)
    model.eval()
    pseudo_data = []
    k_best_lists = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Generating pseudo data"):
        source_ids = batch[0]
        source_ids = source_ids.to(args.device)
        with torch.no_grad():
            preds = model(source_ids) # [batch_size, beam_size, length]
            _k_best_lists = [[tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for ids in batch_ids] for batch_ids in preds.cpu().numpy()]
            k_best_lists.extend(_k_best_lists)

    # Calculating similarity
    for k_best_list, ex in zip(k_best_lists, eval_examples):
        scores = [smooth_bleu.bleu([ex.target], pred)[0] for pred in k_best_list] 
        index = np.argmax(scores)
        y_hat = k_best_list[index]
        pseudo_data.append((ex.source, y_hat))

    objs = [{'code_tokens': source.strip().split(),
             'docstring_tokens': nl.strip().split()}
                    for (source, nl) in pseudo_data]
    output_filepath = os.path.join(args.output_dir, "pseudo_data.jsonl")
    with open(output_filepath, "w") as f:
        for obj in objs:
            json.dump(obj, f)
            f.write('\n')

if __name__ == "__main__":
    main()
