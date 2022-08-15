export CUDA_DEVICE_ORDER=PCI_BUS_ID





seed=3407


model=ViT-MoE-S
model=ViT-MoE-M
#model=ViT-S
#model=ViT-S
#model=Mixer-S


model_name=nominal

experiment=ver3-test-moe
#run=test1
mode=train
#mode=eval-test
#run_name=2022-0815-23:13

#XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 \
python GoB/run.py --config config/config.yaml \
    --experiment ${experiment} \
    --model ${model} \
    --mode ${mode} \
    --seed ${seed} \
    --model_name ${model_name}
#    --run_name ${run_name}



