
python train.py ucf101 RGB data/UCF-101/train.list data/UCF-101/val.list \
       --pretrained_model pretrained/ECO_full_k400_paddle.pdparams \
       --arch ECOfull --num_segments 24 --pretrained_parts finetune --lr 0.001 --clip-gradient 50 \
       --epochs 30 -b 16 -j 4 --dropout 0 --save_dir checkpoint --num_saturate 5 \
       --save_name ECO_full_model_24f_final --log_path log