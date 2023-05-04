import os
from glob import glob
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/cm/shared/hungtq29/self_improvement/CodeBERT/UniXcoder/downstream-tasks/code-generation/pseudo_data_generation")
    parser.add_argument("--output_dir", type=str, default="/cm/shared/hungtq29/datasets/concode_merged")

    return parser.parse_args()

def main():
    args = parse_args()
    for metrics in ["bleu", "codebleu"]:
        # filepaths = glob(os.path.join(args.data_dir, "batch*", metrics, "pseudo_data.json"))
        filepaths = [os.path.join(args.data_dir, "batch{}".format(str(i)), metrics, "pseudo_data.json") for i in range(10)]
        print(filepaths)
        filepaths.sort()
        output_dir = os.path.join(args.output_dir, metrics)
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, "pseudo_data.json")
        with open(output_filepath, "w") as f:
            for filepath in filepaths:
                with open(filepath, encoding="utf-8") as f1:
                    for idx, line in enumerate(f1):
                        line = line.strip()
                        js = json.loads(line)
                        json.dump(js, f)
                        f.write("\n")

if __name__ == "__main__":
    main()
