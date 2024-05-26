visualize predictions of trained model

`python scripts_method/extract_predicts.py --setup p2 --method arctic_sf --load_ckpt ./logs/3d6655b39/checkpoints/last.ckpt --run_on val --extraction_mode vis_pose`

eval_pose `python scripts_method/extract_predicts.py --setup p2 --method arctic_sf --load_ckpt ./logs/3d6655b39/checkpoints/last.ckpt --run_on val --extraction_mode eval_pose`

evaluate metrics: `python scripts_method/evaluate_metrics.py --eval_p logs/3d6655b39/eval --split val --setup p2 --task pose`

`python scripts_method/visualizer.py --exp_folder ./logs/3d6655b39 --seq_name s05_ketchup_use_01_0 --mode pred_mesh --headless`

`python scripts_method/visualizer.py --exp_folder ./logs/3d6655b39 --seq_name s05_ketchup_use_01_0 --mode gt_mesh --headless`

visualize predictions of trained model with coap

vis_pose `python scripts_method/extract_predicts.py --setup p2 --method arctic_sf --load_ckpt ./logs/1_0xxxxxx/checkpoints/last.ckpt --run_on val --extraction_mode vis_pose`

eval_pose `python scripts_method/extract_predicts.py --setup p2 --method arctic_sf --load_ckpt ./logs/1_0xxxxxx/checkpoints/last.ckpt --run_on val --extraction_mode eval_pose`

evaluate metrics: `python scripts_method/evaluate_metrics.py --eval_p logs/1_0xxxxxx/eval --split val --setup p2 --task pose`

`python scripts_method/visualizer.py --exp_folder ./logs/1_0xxxxxx --seq_name s05_ketchup_use_01_0 --mode pred_mesh --headless`

`python scripts_method/visualizer.py --exp_folder ./logs/1_0xxxxxx --seq_name s05_ketchup_use_01_0 --mode gt_mesh --headless`

visualize predictions of pretrained model

`python scripts_method/extract_predicts.py --setup p2 --method arctic_sf --load_ckpt /media/sofie-kamber/EFP_Studium/arctic/logs/28bf3642f/checkpoints/last.ckpt --run_on val --extraction_mode vis_pose`

`python scripts_method/visualizer.py --exp_folder /media/sofie-kamber/EFP_Studium/arctic/logs/28bf3642f --seq_name s05_box_grab_01_0 --mode pred_mesh --headless`

`python scripts_method/visualizer.py --exp_folder /media/sofie-kamber/EFP_Studium/arctic/logs/28bf3642f --seq_name s05_box_grab_01_0 --mode gt_mesh --headless`

`python scripts_method/visualizer.py --exp_folder /media/sofie-kamber/EFP_Studium/arctic/logs/28bf3642f --seq_name s05_microwave_use_02_0 --mode pred_mesh --headless`

train model
`python scripts_method/train.py --setup p2 --method arctic_sf --trainsplit train --valsplit minival --mute --batch_size 64 --eval_every_epoch 1 --num_epoch 30 --exp_key 123456789`