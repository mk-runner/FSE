CUDA_VISIBLE_DEVICES=1 python main_finetune.py \
--task pretrain \
--mimic_cxr_ann_path "/home/miao/data/Code/MSC-V1212-ablation-study/knowledge_encoder/mimic_cxr_annotation_sen.json" \
--data_name mimic_cxr \
--version cxr_pretrain \
--max_seq_len 100 \
--optim AdamW \
--batch_size 32 \
--align_type keywords \
--epochs 100 \
--lr 5.0e-5