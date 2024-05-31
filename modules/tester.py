import logging
import os
from abc import abstractmethod
import cv2
import pandas as pd
import torch
from modules.utils import generate_heatmap


class BaseTester(object):
    def __init__(self, model, metric_ftns, args, task):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.metric_ftns = metric_ftns
        self._load_checkpoint(args['load'])

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)


class Tester(BaseTester):
    def __init__(self, model, metric_ftns, args, test_dataloader, logger, task, runner):
        super(Tester, self).__init__(model, metric_ftns, args, task)
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.runner = runner

    def test(self):
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, test_images_ids = [], [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, sk_ids, sk_masks) in enumerate(
                    self.test_dataloader):
                images = images.to(self.device)
                reports_ids, reports_masks = reports_ids.to(self.device), reports_masks.to(self.device)
                gen_texts, gt_texts = self.model(images, reports_ids, reports_masks, sk_ids, sk_masks, mode='sample')
                test_res.extend(gen_texts)
                test_gts.extend(gt_texts)
                test_images_ids.extend(images_id)
            test_met = self.metric_ftns(gts=test_gts, res=test_res, args=self.args)
            logg_info = ''
            for k, v in test_met.items():
                logg_info += f"{k}: {v}; "
            self.logger.info(f"test metrics: {logg_info}")
            print(f"test metrics: {logg_info}")

            # save the metrics and the predict results
            temp_ids, temp_test_gts, temp_test_res = list(test_met.keys()), [None] * len(test_met), list(
                test_met.values())
            temp_ids.extend(test_images_ids)
            temp_test_gts.extend(test_gts)
            temp_test_res.extend(test_res)
            cur_test_ret = pd.DataFrame({'images_id': temp_ids, 'ground_truth': temp_test_gts,
                                         f'pred_report': temp_test_res})
            cur_test_ret['images_id'] = cur_test_ret['images_id'].apply(lambda x: x.split('_')[-1])
            test_pred_path = os.path.join(self.args['result_dir'], 'test_prediction.csv')
            cur_test_ret.to_csv(test_pred_path, index=False)

    def plot(self):
        assert self.args['batch_size'] == 1 and self.args['beam_size'] == 1
        self.logger.info('Start to plot attention weights in the test set.')
        os.makedirs(os.path.join(self.args['result_dir'], "attentions"), exist_ok=True)
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        mean = mean[:, None, None]
        std = std[:, None, None]

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks, sk_ids, sk_masks) in enumerate(self.test_dataloader):
                images = images.to(self.device)
                reports_ids, reports_masks = reports_ids.to(self.device), reports_masks.to(self.device)
                gen_texts, _ = self.model(images, reports_ids, reports_masks, sk_ids, sk_masks, mode='sample')
                image = torch.clamp((images[0].cpu() * std + mean) * 255, 0, 255).int().cpu().numpy()
                attention_weights = [layer.src_attn.attn.cpu().numpy()[:, :, :-1].mean(0).mean(0) for layer in
                                     self.model.text_decoder.model.decoder.layers]
                for layer_idx, attns in enumerate(attention_weights):
                    assert len(attns) == len(gen_texts)
                    for word_idx, (attn, word) in enumerate(zip(attns, gen_texts)):
                        os.makedirs(os.path.join(self.args['result_dir'], "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx)), exist_ok=True)

                        heatmap = generate_heatmap(image, attn)
                        cv2.imwrite(os.path.join(self.args['result_dir'], "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx), "{:04d}_{}.png".format(word_idx, word)),
                                    heatmap)
