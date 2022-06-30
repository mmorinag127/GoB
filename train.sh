export CUDA_DEVICE_ORDER=PCI_BUS_ID







model=CNN-64
#model=Test
model=ViT-VT/16
model=Mixer-VT/16
model=gMLP-VT/16
#model=GoB-S/16

model_name=nominal

experiment=FiLM
run=test1
mode=train
#mode=eval-test


#XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 \
python GoB/run.py --config config/config.yaml \
    --experiment ${experiment} \
    --model ${model} \
    --mode ${mode} \
    --seed 3407 \
    --model_name ${model_name}



