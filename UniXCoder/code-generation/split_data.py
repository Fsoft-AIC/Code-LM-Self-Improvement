import os
import json
import argparse
from tqdm import tqdm
import numpy as np
NUM_SAMPLES = 100000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/cm/shared/hungtq29/datasets/concode/")
    parser.add_argument("--output_dir", type=str, default="/cm/shared/hungtq29/datasets/concode_splitted")
    parser.add_argument("--num_chunks", type=int, default=10)

    return parser.parse_args()

def main():
    args = parse_args()
 
    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_filepaths = [os.path.join(output_dir, "batch{}.jsonl".format(str(i))) for i in range(args.num_chunks)]
    write_streams = [open(output_filepath, "w") for output_filepath in output_filepaths]
    train_filename = os.path.join(args.data_dir, "train.json")
    batch_size = int(np.ceil(NUM_SAMPLES/args.num_chunks))

    with open(train_filename, encoding="utf-8") as f:
        for idx, line in tqdm(enumerate(f)):
            write_stream  = write_streams[idx // batch_size]
            line = line.strip()
            js = json.loads(line)
            json.dump(js, write_stream)
            write_stream.write("\n")

    for stream in write_streams:
        stream.close()

if __name__ == "__main__":
    main()
