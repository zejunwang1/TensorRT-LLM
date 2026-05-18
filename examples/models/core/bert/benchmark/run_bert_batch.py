# Developed by Wang Zejun
import argparse
import json
import os
import time

import torch
import tensorrt as trt

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._utils import trt_dtype_to_torch
from tensorrt_llm.runtime import Session, TensorInfo
from transformers import AutoTokenizer


OUTPUT_NAME_MAPPING = {
    'BertModel': 'hidden_states',
    'BertForMaskedLM': 'predictions',
    'BertForQuestionAnswering': 'logits',
    'BertForSequenceClassification': 'logits',
    'RobertaModel': 'hidden_states',
    'RobertaForQuestionAnswering': 'logits',
    'RobertaForSequenceClassification': 'logits'
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--engine_dir', type=str, required=True)
    parser.add_argument('--hf_model_dir', type=str, required=True)
    parser.add_argument('--text_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--remove_input_padding', action='store_true')
    return parser.parse_args()


def prepare_inputs(text, tokenizer, remove_padding):
    if remove_padding:
        # Remove padding
        inputs_without_padding = tokenizer(texts)

        input_ids_list = []
        input_lengths = []
        token_type_ids_list = []
        position_ids_list = []
        max_input_length = 0
        for input_ids, token_type_ids in zip(inputs_without_padding['input_ids'], 
                                             inputs_without_padding['token_type_ids']):
            input_len = len(input_ids)
            input_lengths.append(input_len)
            input_ids = torch.tensor(input_ids, dtype=torch.int32)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.int32)
            position_ids = torch.arange(0, input_len, dtype=torch.int32)
            input_ids_list.append(input_ids)
            token_type_ids_list.append(token_type_ids)
            position_ids_list.append(position_ids)
            max_input_length = max(max_input_length, input_len)

        # [num_tokens]
        input_ids = torch.concat(input_ids_list).cuda()
        token_type_ids = torch.concat(token_type_ids_list).cuda()
        position_ids = torch.concat(position_ids_list).cuda()
        input_lengths = torch.tensor(input_lengths, dtype=torch.int32).cuda()
        max_input_length = torch.empty((max_input_length, ), dtype=torch.int32).cuda()

        inputs = {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "max_input_length": max_input_length
        }
        output_info = session.infer_shapes([
            TensorInfo("input_ids", trt.DataType.INT32, input_ids.shape),
            TensorInfo("input_lengths", trt.DataType.INT32, input_lengths.shape),
            TensorInfo("token_type_ids", trt.DataType.INT32, token_type_ids.shape),
            TensorInfo("position_ids", trt.DataType.INT32, position_ids.shape),
            TensorInfo("max_input_length", trt.DataType.INT32, max_input_length.shape)
        ])

    else:
        # Padding
        inputs_with_padding = tokenizer(texts, padding=True)
        input_ids = torch.tensor(inputs_with_padding['input_ids'], dtype=torch.int32).cuda()
        input_lengths = [ sum(x) for x in inputs_with_padding['attention_mask'] ]
        input_lengths = torch.tensor(
            input_lengths, device=input_ids.device, dtype=torch.int32)
        token_type_ids = torch.tensor(
            inputs_with_padding['token_type_ids'], device=input_ids.device, dtype=torch.int32)

        inputs = {
            'input_ids': input_ids,
            'input_lengths': input_lengths,
            'token_type_ids': token_type_ids,
        }
        output_info = session.infer_shapes([
            TensorInfo('input_ids', trt.DataType.INT32, input_ids.shape),
            TensorInfo('input_lengths', trt.DataType.INT32,
                       input_lengths.shape),
            TensorInfo('token_type_ids', trt.DataType.INT32,
                       token_type_ids.shape)
        ])

    outputs = {
        t.name:
        torch.empty(tuple(t.shape),
                    dtype=trt_dtype_to_torch(t.dtype),
                    device='cuda')
        for t in output_info
    }
    return inputs, outputs


if __name__ == '__main__':
    args = parse_arguments()

    tensorrt_llm.logger.set_level(args.log_level)

    config_path = os.path.join(args.engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    remove_padding = config['build_config']['plugin_config']['remove_input_padding']
    assert args.remove_input_padding == remove_padding, \
        f"The engine is build with remove_input_padding={remove_padding}, \
        but the inference runtime is performed with remove_input_padding={args.remove_input_padding}!"

    model_name = config['pretrained_config']['architecture']
    output_name = OUTPUT_NAME_MAPPING[model_name]

    torch.cuda.set_device(0)

    serialize_path = os.path.join(args.engine_dir, 'rank0.engine')
    logger.info(f'Loading engine from {serialize_path}')
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    logger.info(f'Creating session from engine')
    session = Session.from_serialized_engine(engine_buffer)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir)

    texts = []
    with open(args.text_file, mode='r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text:
                texts.append(text)

    # Warmup
    warmup_texts = texts[0 : args.batch_size + 1]
    warmup_inputs, warmup_outputs = prepare_inputs(warmup_texts, tokenizer, remove_padding)
    logger.info(f"Warmup...")
    ok = session.run(inputs=warmup_inputs, outputs=warmup_outputs, stream=0)
    assert ok, "Runtime execution failed"

    n = len(texts)
    batch_size = args.batch_size
    num_batches = int((n - 1) / batch_size) + 1
    logger.info(f"Rank0 is running inference...")
    tic = time.time()
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)
        batch_texts = texts[start : end]
        inputs, outputs = prepare_inputs(batch_texts, tokenizer, remove_padding)
        ok = session.run(inputs=inputs, outputs=outputs, stream=0)

    torch.cuda.synchronize()
    toc = time.time()
    last_batch_res = outputs[output_name]
    print(last_batch_res)
    print('time usage: {}s'.format(toc - tic))

