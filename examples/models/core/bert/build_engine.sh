export hf_model_dir='bert-base-chinese'
export model_name='bert_mlm'
export model='BertForMaskedLM'
export dtype='float16'

# convert
python convert_checkpoint.py \
    --model $model \
    --model_dir $hf_model_dir \
    --output_dir ${model_name}_${dtype}_tllm_checkpoint \
    --dtype $dtype

# build TensorRT engine
trtllm-build --checkpoint_dir ./${model_name}_${dtype}_tllm_checkpoint \
    --output_dir=${model_name}_engine_outputs \
    --remove_input_padding=enable \
    --bert_attention_plugin=${dtype} \
    --max_batch_size 16 \
    --max_input_len 512

