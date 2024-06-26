CUDA_VISIBLE_DEVICES=1 python main_finetune.py \
--task test \
--data_name iu_xray \
--iu_xray_ann_path "iu_xray_annotation_sen_best_reports_keywords_20.json" \
--version iu_top20_temp \
--max_seq_len 60 \
--freeze_text_encoder \
--freeze_image_encoder \
--sk_type keywords \
--sk_topk 20 \
--sk_fusion_strategy cat \
--optim RAdam \
--batch_size 16 \
--lr 1.0e-4 \
--epochs 30 \
--load "iu_xray/finetune_model_best.pth"