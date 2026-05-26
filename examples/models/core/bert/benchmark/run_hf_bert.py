# Developed by Wang Zejun
import argparse
import time

import torch
from transformers import AutoTokenizer, AutoConfig
from transformers import BertModel, BertForMaskedLM, BertForQuestionAnswering, BertForSequenceClassification


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_dir', type=str, required=True)
    parser.add_argument('--text_file', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    torch.cuda.set_device(0)

    config = AutoConfig.from_pretrained(args.hf_model_dir)
    model_name = config.architectures[0]
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir)
    model = globals()[f'{model_name}'].from_pretrained(args.hf_model_dir).cuda().to(torch.float16).eval()

    texts = []
    with open(args.text_file, mode='r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text:
                texts.append(text)

    inputs = tokenizer(texts, padding=True, return_tensors='pt').to('cuda')

    # Warmup
    with torch.no_grad():
        warmup_res = model(**inputs)

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    torch.cuda.synchronize()
    end = time.time()
    print(outputs[0])
    print('time usage: {}s'.format(end - start))

