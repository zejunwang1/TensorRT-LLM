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
    parser.add_argument('--batch_size', type=int, default=64)
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

    # Warmup
    print('Warmup...')
    warm_texts = texts[0 : args.batch_size]
    warm_inputs = tokenizer(warm_texts, padding=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
        warmup_outputs = model(**warm_inputs)

    n = len(texts)
    batch_size = args.batch_size
    num_batches = int((n - 1) / batch_size) + 1
    print('Running inference...')
    tic = time.time()
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)
        batch_texts = texts[start : end]
        inputs = tokenizer(batch_texts, padding=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = model(**inputs)

    torch.cuda.synchronize()
    toc = time.time()
    print(outputs[0])
    print('time usage: {}s'.format(toc - tic))

