export CUDA_DEVICE_ORDER=PCI_BUS_ID





seed=3407


model=Mixer-S-4
model=GoB-S-4

head=MLP-S-1
head=MLP-S-4


moe=Token-MoE-2-4-1-4
#moe=MoE-2-4-1-1
moe=None

model_name=nominal


experiment=GoB-test
#run=test1
mode=train
mode=eval-test
run_name=2022-0914-12:00


#XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 \
CUDA_VISIBLE_DEVICES=2 \
python GoB/run.py \
    --n_device 1 \
    --config config/config.yaml \
    --experiment ${experiment} \
    --model ${model} \
    --moe ${moe} \
    --head ${head} \
    --mode ${mode} \
    --seed ${seed} \
    --model_name ${model_name} \
    --run_name ${run_name}
#    --isMoE 



# nohup sh train.sh >/dev/null 2>&1 & 