CUDA_VISIBLE_DEVICES=1 python main_finetune.py \
--task test \
--data_name mimic_cxr \
--mimic_cxr_ann_path "/home/miao/data/Code/MSC-V1212-ablation-study/knowledge_encoder/mimic_cxr_annotation_sen_best_reports_keywords_20.json" \
--version mimic_cxr_top5 \
--max_seq_len 100 \
--freeze_text_encoder \
--freeze_image_encoder \
--sk_type keywords \
--sk_topk 5 \
--sk_fusion_strategy cat \
--optim RAdam \
--batch_size 16 \
--lr 5.0e-5 \
--epochs 50 \
--load "/home/miao/data/Code/results/ablation study/top5-64-gpu32-new/model_best.pth"