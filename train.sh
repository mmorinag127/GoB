export CUDA_DEVICE_ORDER=PCI_BUS_ID


export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}://data/morinaga/cache/pypoetry/virtualenvs/gob-gyPZHm3D-py3.8/lib/python3.8/site-packages/tensorrt/


seed=3407


model=ViT-L:8-D:32-NH:64-EXP:4-WI:ZerO
head=MLP-L:1-D:32-WI:ZerO

# model=FvT-L:4-D:32-NH:4-EXP:4-WI:ZerO
# head=MLP-L:1-D:32-WI:ZerO

# model=ViT-L:4-D:32-NH:4-EXP:4-WI:ZerO
# head=MLP-L:1-D:32-WI:ZerO

# model=Test
# head=Test

MoE=Token-CYC:2-NE:4-K:4-C:4
MoE=None
FiLM=None

model_name=nominal


experiment=LargeR-64-v5
#experiment=LargeR-64-v6
# experiment=Test2

mode=train 
mode=eval-test



run_name=test-wd1e_5-lr1e_4
run_name=test3


#XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 \
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#RAY_IGNORE_UNHANDLED_ERRORS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
RAY_IGNORE_UNHANDLED_ERRORS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python GoB/run_largeR.py \
    --config config/config-largeR.yaml \
    --experiment ${experiment} \
    --model ${model} \
    --moe ${MoE} \
    --film ${FiLM} \
    --head ${head} \
    --mode ${mode} \
    --seed ${seed} \
    --model_name ${model_name} \
    --run_name ${run_name}
#    --isMoE 

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 ray start --head


# nohup sh train.sh >/dev/null 2>&1 & 