import os

# lang = ["go",  "java",  "javascript",  "php",  "python",  "ruby"]
lang = "python"
lang = "java"
lang = "javascript"
lang = "go"

CHECKPOINT_DIR_PATH="/home/rizwan/DPR_models/biencoder_models/"+lang
pretrained_model="microsoft/codebert-base"


DEVICES=[0,1,2,3,4,5,6,7]
CUDA_VISIBLE_DEVICES=','.join([str(i) for i in DEVICES])

command = "CUDA_VISIBLE_DEVICES="+CUDA_VISIBLE_DEVICES+" python -m torch.distributed.launch --nproc_per_node="+str(len(DEVICES))+" train_dense_encoder.py \
 --max_grad_norm 2.0 \
 --encoder_model_type hf_roberta \
 --pretrained_model_cfg " + pretrained_model   + " \
 --eval_per_epoch 1 \
 --seed 12345 \
 --sequence_length 256 \
 --warmup_steps 1237 \
 --batch_size 8 \
 --train_file /home/rizwan/CodeBERT/data/codesearch/train_valid/"+lang+"/train.txt   \
 --dev_file /home/rizwan/CodeBERT/data/codesearch/train_valid/"+lang+"/valid.txt \
 --output_dir "+CHECKPOINT_DIR_PATH+" \
 --learning_rate 2e-5 \
 --num_train_epochs 5 \
 --dev_batch_size 16 \
 --val_av_rank_start_epoch 1 \
 --fp16 \
"
print (command, flush=True)
os.system(command)






