import os
import json
import torch
import random
import argparse
import numpy as np

import yaml
from modules.tokenizers_new import build_my_tokenizer
from modules.dataloaders import PretrainLoader, FinetuneLoader, PretrainInferenceLoader, PretrainInferenceLoaderMIMICOne
from modules.metrics.metrics import compute_all_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer_finetune_iu import PTrainer, FTrainer, PretrainTester, Tester
from modules.utils import PretrainTestAnalysis, SetLogger, setup_seed
from models.model_pretrain_region_knowledge import Pretrain
from models.model_pretrain_region_knowledge_local import LocalPretrain
from models.model_pretrain_region_knowledge_global import GlobalPretrain
from models.model_pretrain_region_knowledge_inference_iu import PretrainInference
from models.model_finetune_region_knowledge_v1121 import FineTune

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import wandb
os.environ["WANDB_API_KEY"] = '****************'
os.environ["WANDB_MODE"] = "offline"


def main():
    # -------------------------------
    # load hyper-param
    # -------------------------------
    parse = argparse.ArgumentParser()
    # basic configuration
    # pretrain: cross-modal alignment module
    # pretrain_inference: historical similar cases retrieval for each image,
    #                     forming mimic_cxr_annotation_sen_best_reports_keywords_20.json
    # finetune: train text decoder based on historical similar cases
    # test: text generation for test dataset
    parse.add_argument('--task', type=str, default='finetune',
                       choices=['pretrain', 'pretrain_inference', 'finetune', 'test'])
    # data configuration
    parse.add_argument('--data_name', type=str, choices=['mimic_cxr', 'iu_xray'], default='iu_xray')
    parse.add_argument('--mimic_cxr_ann_path', type=str)
    parse.add_argument('--iu_xray_ann_path', type=str)
    parse.add_argument('--text_decoder', type=str, choices=['r2gen', 'bert', 'cmn'], default='r2gen')
    parse.add_argument('--visual_encoder', type=str, choices=['resnet101', 'ViT-B-32'], default='resnet101')
    parse.add_argument('--tokenizer_model', type=str, choices=['wordlevel', 'wordpiece'], default='wordlevel')
    parse.add_argument('--tokenizer_type', type=str, choices=['uncased', 'cased'], default='uncased')
    parse.add_argument('--max_seq_len', type=int, default=60)
    parse.add_argument('--freeze_image_encoder', action='store_true', help='whether freeze the image encoder')
    parse.add_argument('--freeze_text_encoder', action='store_true', help='whether freeze the text encoder')
    parse.add_argument('--is_save_checkpoint', action='store_true', help='whether save checkpoint')
    # specific knowledge configuration
    parse.add_argument('--sk_type', type=str, choices=['report', 'keywords'], default='keywords')
    parse.add_argument('--sk_topk', type=int, default=5)
    parse.add_argument('--sk_fusion_strategy', type=str, choices=['mean', 'cat'], default='cat')
    parse.add_argument('--sk_fusion_num_layers', type=int, default=1)
    parse.add_argument('--sk_file_name', type=str, default='_best_reports_keywords_')
    # trainer configuration
    parse.add_argument('--optim', type=str, choices=['AdamW', 'RAdam', "Adam"], default='RAdam',
                       help='in the first stage, the optimizer is AdamW with lr=5e-5, '
                            'in the second stage, optimizer is RAdam with lr=5e-5')
    parse.add_argument('--lr_scheduler', type=str, choices=['StepLR', 'ReduceLROnPlateau'],
                       default='ReduceLROnPlateau')
    parse.add_argument('--lr', type=float, default=5.0e-5)  # 5.0e-5
    parse.add_argument('--ft_monitor_metric', type=str, default='RCB')  # choices={metrics, RC, RB, RCB}
    parse.add_argument('--epochs', type=int, default=100)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parse.add_argument('--load', type=str, help='whether to load the pre-trained model.')
    parse.add_argument('--version', type=str, default='long_sentence', help='the name of experiment')
    # sk_type and align_type is the same.
    parse.add_argument('--align_type', type=str, choices=['report', 'keywords'], default='keywords')
    parse.add_argument('--align_loss', type=str, choices=['local', 'global', 'multi-level'], default='multi-level')
    cmd = parse.parse_args()
    cmd.config = 'config/finetune_config.yaml'
    args = yaml.load(open(cmd.config), Loader=yaml.FullLoader)
    cmd = vars(cmd)
    args.update(cmd)
    args['image_dir'] = args[f'{args["data_name"]}_image_dir']
    args['ann_path'] = args[f'{args["data_name"]}_ann_path']
    args['text_decoder'] = args['text_decoder'].lower()
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    args['result_dir'] = f'{args["result_dir"]}/{args["data_name"]}/{args["task"]}/{args["version"]}'
    os.makedirs(args['result_dir'], exist_ok=True)

    logger = SetLogger(f'{args["result_dir"]}/{args["task"]}_{args["text_decoder"]}_{args["sk_topk"]}.log', 'a')
    if args['task'] in ['pretrain', 'pretrain_inference']:
        args['monitor_mode'] = args['pt_monitor_mode']
        args['monitor_metric'] = args['pt_monitor_metric']
        args['lr_monitor_metric'] = args['pt_lr_monitor_metric']
    else:
        args['monitor_mode'] = args['ft_monitor_mode']
        args['monitor_metric'] = args['ft_monitor_metric']
        args['lr_monitor_metric'] = args['ft_lr_monitor_metric']
    # -------------------------------
    # init wandb
    runner = wandb.init(
        project=f'rrg_{args["data_name"]}_{args["task"]}_{args["text_decoder"]}_{args["sk_topk"]}',
        config=args,
    )
    # -------------------------------
    # fix random seeds
    # -------------------------------
    setup_seed(args["seed"])
    # -------------------------------
    logger.info('start load data...')
    # -------------------------------
    # create tokenizer
    # -------------------------------
    print("load tokenizer...")
    tokenizer = build_my_tokenizer(tokenizer_dir=args['tokenizer_dir'], model=args['tokenizer_model'],
                                   data_name=args['data_name'], ann_path=args['ann_path'],
                                   tokenizer_type=args['tokenizer_type'], is_same_tokenizer=True)
    args['vocab_size'] = tokenizer.get_vocab_size()
    args['suppress_UNK'] = tokenizer.token_to_id('[UNK]')  # used for the CMN or r2gen text decoder
    # -------------------------------
    # save the config
    params = ''
    for key, value in args.items():
        params += f'{key}:\t{value}\n'
    logger.info(params)
    print(params)
    # -------------------------------
    # create data loader
    # -------------------------------
    mimic_train_loader = None
    if args['task'] == 'pretrain':
        train_dataloader = PretrainLoader(args, tokenizer, split='train', shuffle=False, drop_last=False)
        val_dataloader = PretrainLoader(args, tokenizer, split='val', shuffle=False, drop_last=False)
        test_dataloader = PretrainLoader(args, tokenizer, split='test', shuffle=False, drop_last=False)
    elif args['task'] == 'pretrain_inference':
        mimic_train_loader = PretrainInferenceLoaderMIMICOne(args, split='train', shuffle=False, drop_last=False)
        train_dataloader = PretrainInferenceLoader(args, split='train', shuffle=False, drop_last=False)
        val_dataloader = PretrainInferenceLoader(args, split='val', shuffle=False, drop_last=False)
        test_dataloader = PretrainInferenceLoader(args, split='test', shuffle=False, drop_last=False)
    elif args['task'] == 'finetune':
        train_dataloader = FinetuneLoader(args, tokenizer, split='train', shuffle=False, drop_last=False)
        val_dataloader = FinetuneLoader(args, tokenizer, split='val', shuffle=False, drop_last=False)
        test_dataloader = FinetuneLoader(args, tokenizer, split='test', shuffle=False, drop_last=False)
    else:  # test
        train_dataloader = None
        val_dataloader = None
        test_dataloader = FinetuneLoader(args, tokenizer, split='test', shuffle=False, drop_last=False)

    print(f"train_data is {len(train_dataloader.dataset) if train_dataloader is not None else 'None'}, "
          f"val_data is {len(val_dataloader.dataset) if val_dataloader is not None else 'None'}, "
          f"test_data is {len(test_dataloader.dataset)}")
    logger.info(f"train_data is {len(train_dataloader.dataset) if train_dataloader is not None else 'None'}, "
          f"val_data is {len(val_dataloader.dataset) if val_dataloader is not None else 'None'}, "
          f"test_data is {len(test_dataloader.dataset)}")

    runner.config.update({
        'vocab_size': tokenizer.get_vocab_size(),
        'suppress_UNK': args['suppress_UNK'],
        'train_len': len(train_dataloader.dataset) if train_dataloader is not None else 'None',
        'val_len': len(val_dataloader.dataset) if val_dataloader is not None else "None",
        'test_len': len(test_dataloader.dataset)
    }, allow_val_change=True)
    # -------------------------------
    # build model architecture
    # -------------------------------
    if args['task'] == 'pretrain':
        if args['align_loss'] == 'multi-level':
            model = Pretrain(args, tokenizer, args['data_name'])
        elif args['align_loss'] == 'local':
            model = LocalPretrain(args, tokenizer, args['data_name'])
        else: # global
            model = GlobalPretrain(args, tokenizer, args['data_name'])
    elif args['task'] == 'pretrain_inference':
        model = PretrainInference(args, data_name=args['data_name'])
    else:  # finetune or test
        model = FineTune(args, tokenizer, args['data_name'])
    model = model.to(args['device'])
    runner.watch(model, log='all')
    # -------------------------------
    print(f'finish instantiate model!, Trainable parameters:{str(model).split("Trainable parameters:")[1]}M')
    logger.info(f'finish instantiate model!, Trainable parameters:{str(model).split("Trainable parameters:")[1]}M')
    # get function handles of loss and metrics
    # -------------------------------
    metrics = compute_all_scores
    # -------------------------------
    # build optimizer, learning rate scheduler
    # -------------------------------
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # -------------------------------
    # build trainer and start to train
    logger.info(f'start {args["task"]}!')
    print(f'start {args["task"]}!')
    # -------------------------------
    kwarg = {"model": model, "metric_ftns": metrics, "optimizer": optimizer, "args": args,
             "lr_scheduler": lr_scheduler, "train_dataloader": train_dataloader, "val_dataloader": val_dataloader,
             "test_dataloader": test_dataloader, "logger": logger, "task": args['task'], 'runner': runner,
             'is_save_checkpoint': args['is_save_checkpoint'], 'mimic_train_loader': mimic_train_loader}

    if args['task'] == 'pretrain':
        trainer = PTrainer(**kwarg)
        trainer.train()
    elif args['task'] == 'pretrain_inference':
        # kwarg = {'model': model, 'train_dataloader': train_dataloader, 'val_dataloader': val_dataloader,
        #          'test_dataloader': test_dataloader, 'logger': logger, 'args': args, 'mimic_train_loader': mimic_train_loader}
        tester = PretrainTester(**kwarg)
        save_file_name = args['ann_path'].split('.json')[0] + f'{args["sk_file_name"]}{args["sk_topk"]}.json'
        if args['data_name'] == 'mimic_cxr':
            specific_knowledge_data = tester.predict_mimic_cxr()
            tester.get_specific_knowledge_mimic_cxr(specific_knowledge_data, save_file_name=save_file_name)
        else:
            specific_knowledge_data = tester.predict_iu_xray()
            tester.get_specific_knowledge_iu_xray(specific_knowledge_data, save_file_name=save_file_name)
    elif args["task"] == 'finetune':
        trainer = FTrainer(**kwarg)
        trainer.train()
    else:   # inference
        trainer = Tester(**kwarg)
        trainer.test()
    runner.finish()


if __name__ == '__main__':
    main()
