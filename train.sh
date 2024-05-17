#!/bin/bash
set -e

echo "Training model"
export COMET_API_KEY="loY9fcwoXWm2ktvXs9jX0Sxfu"
export COMET_WORKSPACE="sofiekamber"
python scripts_method/train.py --setup p2 --method arctic_sf --trainsplit train --valsplit minival --mute --batch_size 64 --eval_every_epoch 1 --num_epoch 30 --exp_key 3d6655b39

# ID 3d6655b39

#[-h] [--method {None,arctic_sf,arctic_lstm,field_sf,field_lstm}] [--exp_key EXP_KEY] [--extraction_mode EXTRACTION_MODE] [--img_feat_version IMG_FEAT_VERSION] [--window_size WINDOW_SIZE] [--eval] [--agent_id AGENT_ID]
#                [--load_from LOAD_FROM] [--load_ckpt LOAD_CKPT] [--infer_ckpt INFER_CKPT] [--resume_ckpt RESUME_CKPT] [-f] [--trainsplit {None,train,smalltrain,minitrain,tinytrain}] [--valsplit {None,val,smallval,tinyval,minival}]
#                [--run_on RUN_ON] [--setup SETUP] [--log_every LOG_EVERY] [--eval_every_epoch EVAL_EVERY_EPOCH] [--lr_dec_epoch LR_DEC_EPOCH [LR_DEC_EPOCH ...]] [--num_epoch NUM_EPOCH] [--lr LR] [--lr_dec_factor LR_DEC_FACTOR]
#                [--lr_decay LR_DECAY] [--num_exp NUM_EXP] [--acc_grad ACC_GRAD] [--batch_size BATCH_SIZE] [--test_batch_size TEST_BATCH_SIZE] [--num_workers NUM_WORKERS] [--eval_on {None,val,test,minival,minitest}] [--mute] [--no_vis]
#                [--cluster] [--cluster_node CLUSTER_NODE] [--bid BID]
