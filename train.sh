export CUDA_DEVICE_ORDER=PCI_BUS_ID





seed=3407


model=Mixer-S-4

head=MLP-S-1
head=MLP-S-4


moe=Token-MoE-2-4-1-4
#moe=MoE-2-4-1-1
moe=None

FiLM=None
FiLM=FiLM-4-4
# FiLM=FiLM-4-8
# FiLM=FiLM-4-16
# FiLM=FiLM-4-32
#FiLM=FiLM-8-32
# FiLM=FiLM-16-32
FiLM=FiLM-8-128v2

model_name=nominal


experiment=FiLM-test-4
#experiment=test-grad
#run=test1
mode=train
mode=eval-test
run_name=2022-10:06-10:00
#run_name=2022-10:07-10:00


#XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 \
CUDA_VISIBLE_DEVICES=0 \
python GoB/run.py \
    --n_device 1 \
    --config config/config.yaml \
    --experiment ${experiment} \
    --model ${model} \
    --moe ${moe} \
    --film ${FiLM} \
    --head ${head} \
    --mode ${mode} \
    --seed ${seed} \
    --model_name ${model_name} \
    --run_name ${run_name}
#    --isMoE 



# nohup sh train.sh >/dev/null 2>&1 & 