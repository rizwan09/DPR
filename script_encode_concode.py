import os

lang='java'
CHECKPOINT="/home/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/dpr_biencoder.5.1786"
pretrained_model="microsoft/codebert-base"

FILE = "/home/wasiahmad/workspace/projects/CodeBART/data/codeXglue/text-to-code/concode/train.json"
OUTPUT_ENCODED_FILE = "/local/rizwan/DPR_models/github_encoddings_conode/"+lang+"/concode_train.json"

DEVICES = [0,1,2,3,4,5,6,7]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])

command = "CUDA_VISIBLE_DEVICES=" + CUDA_VISIBLE_DEVICES + \
          ' python  -m torch.distributed.launch --nproc_per_node='+str(len(DEVICES)) + \
          ' generate_dense_embeddings.py ' \
          ' --model_file  '+ CHECKPOINT + \
          ' --encoder_model_type hf_roberta ' \
          ' --pretrained_model_cfg   ' + pretrained_model + \
          ' --batch_size 128 --ctx_file  ' + FILE + \
          ' --shard_id 0 ' \
          ' --dataset CONCODE ' \
          ' --num_shards 1 ' \
          ' --out_file ' + OUTPUT_ENCODED_FILE

print(command, flush=True)
os.system(command)








