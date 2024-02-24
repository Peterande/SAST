from typing import Any, Optional, Tuple, Union, Dict
from warnings import warn

import numpy as np
import pytorch_lightning as pl
import torch
import torch as th
import torch.distributed as dist
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT

from data.genx_utils.labels import ObjectLabels
from data.utils.types import DataType, LstmStates, ObjDetOutput, DatasetSamplingMode, BackboneFeatures
from models.detection.yolox.utils.boxes import postprocess
from models.detection.yolox_extension.models.detector import YoloXDetector
from utils.evaluation.prophesee.evaluator import PropheseeEvaluator
from utils.evaluation.prophesee.io.box_loading import to_prophesee
from utils.padding import InputPadderFromShape
from .utils.detection import BackboneFeatureSelector, EventReprSelector, RNNStates, Mode, mode_2_string, \
    merge_mixed_batches
from utils.evaluation.prophesee.visualize.vis_utils import LABELMAP_GEN1, LABELMAP_GEN4_SHORT, draw_bboxes
import pandas as pd
import cv2
import copy
import matplotlib.pyplot as plt


class Module(pl.LightningModule):
    def __init__(self, full_config: DictConfig):
        super().__init__()

        self.full_config = full_config

        self.mdl_config = full_config.model
        in_res_hw = tuple(self.mdl_config.backbone.in_res_hw)
        self.input_padder = InputPadderFromShape(desired_hw=in_res_hw)

        self.mdl = YoloXDetector(self.mdl_config)
        self.dataframe = pd.DataFrame()
        self.imgs = []
        self.tokens = []
        self.img_tokens = []
        self.id = 1
        self.id2 = 0
        self.plt_num = 0

        self.mode_2_rnn_states: Dict[Mode, RNNStates] = {
            Mode.TRAIN: RNNStates(),
            Mode.VAL: RNNStates(),
            Mode.TEST: RNNStates(),
        }

        # self.automatic_optimization=False

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_name = self.full_config.dataset.name
        self.mode_2_hw: Dict[Mode, Optional[Tuple[int, int]]] = {}
        self.mode_2_batch_size: Dict[Mode, Optional[int]] = {}
        self.mode_2_psee_evaluator: Dict[Mode, Optional[PropheseeEvaluator]] = {}
        self.mode_2_sampling_mode: Dict[Mode, DatasetSamplingMode] = {}

        # self.started_training = True

        dataset_train_sampling = self.full_config.dataset.train.sampling
        dataset_eval_sampling = self.full_config.dataset.eval.sampling
        assert dataset_train_sampling in iter(DatasetSamplingMode)
        assert dataset_eval_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.RANDOM)
        if stage == 'fit':  # train + val
            self.train_config = self.full_config.training
            self.train_metrics_config = self.full_config.logging.train.metrics

            if self.train_metrics_config.compute:
                self.mode_2_psee_evaluator[Mode.TRAIN] = PropheseeEvaluator(
                    dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_psee_evaluator[Mode.VAL] = PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.TRAIN] = dataset_train_sampling
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling

            for mode in (Mode.TRAIN, Mode.VAL):
                self.mode_2_hw[mode] = None
                self.mode_2_batch_size[mode] = None
            self.started_training = False
        elif stage == 'validate':
            mode = Mode.VAL
            self.mode_2_psee_evaluator[mode] = PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        elif stage == 'test':
            mode = Mode.TEST
            self.mode_2_psee_evaluator[mode] = PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.TEST] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        else:
            raise NotImplementedError
        self.iou_loss, self.conf_loss, self.cls_loss = 0, 0, 0
        self.stage1, self.stage2, self.stage3, self.stage4 = 0, 0, 0, 0

    # def forward(self,
    #             event_tensor: th.Tensor,
    #             previous_states: Optional[LstmStates] = None,
    #             retrieve_detections: bool = True,
    #             targets=None) \
    #         -> Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], LstmStates]:
    #     return self.mdl(x=event_tensor,
    #                     previous_states=previous_states,
    #                     retrieve_detections=retrieve_detections,
    #                     targets=targets)

    def forward(self,
                event_tensor: th.Tensor,
                previous_states: Optional[LstmStates] = None) \
            -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        output = self.mdl.forward_backbone(x=event_tensor,
                        previous_states=previous_states)[0]
        output = [output[i] for i in [1, 2, 3, 4]]
        return output
    
    def get_worker_id_from_batch(self, batch: Any) -> int:
        return batch['worker_id']

    def get_data_from_batch(self, batch: Any):
        return batch['data']

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        batch = merge_mixed_batches(batch)
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        mode = Mode.TRAIN
        self.started_training = True
        step = self.trainer.global_step
        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        token_mask_sequence = data.get(DataType.TOKEN_MASK, None)

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        backbone_feature_selector = BackboneFeatureSelector()
        ev_repr_selector = EventReprSelector()
        obj_labels = list()
        P = [0, 0, 0, 0]
        # opt = self.optimizers()
        for tidx in range(sequence_len):
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            if token_mask_sequence is not None:
                token_masks = self.input_padder.pad_token_mask(token_mask=token_mask_sequence[tidx])
            else:
                token_masks = None

            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            backbone_features, states, p, img, tokens_and_scores = self.mdl.forward_backbone(x=ev_tensors,
                                                                  previous_states=prev_states,
                                                                  token_mask=token_masks)
            
            if img is not None and tokens_and_scores[0][0] is not None:
                # Remove Padding
                img = img[:-24, :, :]
                scores = [x[0] for x in tokens_and_scores]
                scores[0] = scores[0][:-6, :, :]
                scores[1] = scores[1][:-3, :, :]
                # scores[2] = scores[2][:-1, :, :]
                tokens = [x[1] for x in tokens_and_scores]
                tokens[0] = tokens[0][:-6, :, :]
                tokens[1] = tokens[1][:-3, :, :]
                # tokens[2] = tokens[2][:-1, :, :]  

                # Resize

                scores[0] = cv2.resize(scores[0], img.shape[-2::-1], interpolation=cv2.INTER_NEAREST)
                scores[1] = cv2.resize(scores[1], img.shape[-2::-1], interpolation=cv2.INTER_NEAREST)
                # scores[2] = cv2.resize(scores[2], img.shape[-2::-1], interpolation=cv2.INTER_NEAREST)

                tokens[0] = cv2.resize(tokens[0], img.shape[-2::-1], interpolation=cv2.INTER_NEAREST)
                tokens[1] = cv2.resize(tokens[1], img.shape[-2::-1], interpolation=cv2.INTER_NEAREST)
                # tokens[2] = cv2.resize(tokens[2], img.shape[-2::-1], interpolation=cv2.INTER_NEAREST)

                horizontal_space_size = 60
                vertical_space_size = 60
                height, width, channels = img.shape
                horizontal_space = 255 * np.ones((height, horizontal_space_size, channels), dtype=img.dtype)
                vertical_space = 255 * np.ones((vertical_space_size, 3 * width + 2 * horizontal_space_size, channels), dtype=img.dtype)

                if img is not None and tokens_and_scores[0][0] is not None:
                                           
                    row1 = np.concatenate([img, horizontal_space, scores[0], horizontal_space, scores[1]], axis=1)
                    row2 = np.concatenate([img, horizontal_space, tokens[0], horizontal_space, tokens[1]], axis=1)
                    rows = np.concatenate([vertical_space, row1, vertical_space, row2], axis=0)
                    self.img_tokens.append(rows)
                    self.plt_num += 1
                    if self.plt_num % 10 == 0:
                        height, width = rows.shape[:2]
                        plt.figure(figsize=(width / 100, height / 100), dpi=100)
                        plt.imshow(rows[:, :, ::-1], aspect='auto')
                        plt.axis('off')
                        plt.savefig('rows_visualization.png', bbox_inches='tight', pad_inches=0)   


            P[0] += (p[0]) / sequence_len
            P[1] += (p[1]) / sequence_len
            P[2] += (p[2]) / sequence_len
            P[3] += (p[3]) / sequence_len
            prev_states = states

            current_labels, valid_batch_indices = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
            # Store backbone features that correspond to the available labels.
            if len(current_labels) > 0:
                backbone_feature_selector.add_backbone_features(backbone_features=backbone_features,
                                                                selected_indices=valid_batch_indices)
                obj_labels.extend(current_labels)
                ev_repr_selector.add_event_representations(event_representations=ev_tensors,
                                                           selected_indices=valid_batch_indices)

        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)
        assert len(obj_labels) > 0
        # Batch the backbone features and labels to parallelize the detection code.
        selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()
        labels_yolox = ObjectLabels.get_labels_as_batched_tensor(obj_label_list=obj_labels, format_='yolox')
        labels_yolox = labels_yolox.to(dtype=self.dtype)

        predictions, losses = self.mdl.forward_detect(backbone_features=selected_backbone_features,
                                                      targets=labels_yolox)

        if self.mode_2_sampling_mode[mode] in (DatasetSamplingMode.MIXED, DatasetSamplingMode.RANDOM):
            # We only want to evaluate the last batch_size samples if we use random sampling (or mixed).
            # This is because otherwise we would mostly evaluate the init phase of the sequence.
            predictions = predictions[-batch_size:]
            obj_labels = obj_labels[-batch_size:]

        pred_processed = postprocess(prediction=predictions,
                                     num_classes=self.mdl_config.head.num_classes,
                                     conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                     nms_thre=self.mdl_config.postprocess.nms_threshold)

        loaded_labels_proph, yolox_preds_proph = to_prophesee(obj_labels, pred_processed)

        assert losses is not None
        assert 'loss' in losses
        # self.trainer._logger_connector.progress_bar_metrics['loss'] = self.smooth_loss(losses['loss'].clone().detach(), 3).item()
        # self.trainer._logger_connector.progress_bar_metrics['iou_loss'] = self.smooth_loss(losses['iou_loss'].clone().detach(), 0).item()
        # self.trainer._logger_connector.progress_bar_metrics['conf_loss'] = self.smooth_loss(losses['conf_loss'].clone().detach(), 1).item()
        # self.trainer._logger_connector.progress_bar_metrics['cls_loss'] = self.smooth_loss(losses['cls_loss'].clone().detach(), 2).item()
        self.smooth_loss(P[0], 1)
        self.smooth_loss(P[1], 2)
        self.smooth_loss(P[2], 3)
        self.smooth_loss(P[3], 4)
        self.trainer._logger_connector.progress_bar_metrics['S'] = (self.stage1 + self.stage2 + self.stage3 + self.stage4) // 1
        self.trainer._logger_connector.progress_bar_metrics['N'] = P[0]  + P[1] + P[2] + P[3]
        self.trainer._logger_connector.progress_bar_metrics['STEPS'] = self.trainer.global_step
        # For visualization, we only use the last batch_size items.
        output = {
            ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-batch_size:],
            ObjDetOutput.PRED_PROPH: yolox_preds_proph[-batch_size:],
            ObjDetOutput.EV_REPR: ev_repr_selector.get_event_representations_as_list(start_idx=-batch_size),
            ObjDetOutput.SKIP_VIZ: False,
            'loss': losses['loss']
        }

        # Logging
        prefix = f'{mode_2_string[mode]}/'
        log_dict = {f'{prefix}{k}': v for k, v in losses.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)

        if mode in self.mode_2_psee_evaluator:
            self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
            self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)
            if self.train_metrics_config.detection_metrics_every_n_steps is not None and \
                    step > 0 and step % self.train_metrics_config.detection_metrics_every_n_steps == 0:
                self.run_psee_evaluator(mode=mode)
        # loss = losses['loss'] + P / 10
        # opt.optimizer.zero_grad()
        # self.manual_backward(loss)
        # opt.step()
        # dict = {}
        # for name, param in self.named_parameters():
        #     if 'to_scores' in name:
        #         print(name, param.grad)
        #         dict[name] = param
        return output

    def _val_test_step_impl(self, batch: Any, mode: Mode) -> Optional[STEP_OUTPUT]:
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        assert mode in (Mode.VAL, Mode.TEST)
        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        AP = ''
        if is_first_sample[0] and self.id2 != 0 and len(self.img_tokens) > 0:
            mode = Mode.TEST
            print('Testing split No ', self.id2)
            self.id2 += 1
            AP = self.run_psee_evaluator(mode=mode, need_return=True)
            self.mode_2_psee_evaluator[mode].reset_buffer()
        elif is_first_sample[0] and self.id2 == 0:
            self.id2 = 1
        if is_first_sample[0] and len(self.img_tokens) > 0:
            if float(AP[0:6]) > 55:
                out = cv2.VideoWriter('vis/videos/4_477_notag/' + str(self.id) + '_' + AP + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, self.img_tokens[0].shape[-2::-1])  # 1是帧率，可以根据需要调整
                for i in range(len(self.img_tokens)):
                    out.write(self.img_tokens[i])
                out.release()
            self.img_tokens = []
            self.id += 1
        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        backbone_feature_selector = BackboneFeatureSelector()
        ev_repr_selector = EventReprSelector()
        obj_labels = list()
        imgs, tokens = [], []

        for tidx in range(sequence_len):
            collect_predictions = (tidx == sequence_len - 1) or \
                                  (self.mode_2_sampling_mode[mode] == DatasetSamplingMode.STREAM)
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]
            
            ratios = ev_tensors.shape[0] * torch.count_nonzero(ev_tensors, dim=[1, 2, 3]) / ev_tensors.numel()
            backbone_features, states, index_lens, img, tokens_and_scores = self.mdl.forward_backbone(x=ev_tensors, previous_states=prev_states)
            # _, _, _, img, tokens_and_scores = self.mdl.forward_backbone(x=ev_tensors, previous_states=None)
            if img is not None and tokens_and_scores[0][0] is not None:
                # Remove Padding
                img = img[:-24, :, :]
                scores = [x[0] for x in tokens_and_scores]
                scores[0] = scores[0][:-6, :, :]
                scores[1] = scores[1][:-3, :, :]
                scores[2] = scores[2][:-1, :, :]

                tokens = [x[1] for x in tokens_and_scores]
                tokens[0] = tokens[0][:-6, :, :]
                tokens[1] = tokens[1][:-3, :, :]
                tokens[2] = tokens[2][:-1, :, :]  

                # Resize
                # img = cv2.resize(img, [img.shape[1] // 2, img.shape[0] // 2], interpolation=cv2.INTER_NEAREST)
                scores[0] = cv2.resize(scores[0], img.shape[-2::-1], interpolation=cv2.INTER_NEAREST)
                scores[1] = cv2.resize(scores[1], img.shape[-2::-1], interpolation=cv2.INTER_NEAREST)
                scores[2] = cv2.resize(scores[2], img.shape[-2::-1], interpolation=cv2.INTER_NEAREST)
                # scores[3] = cv2.resize(scores[3], img.shape[-2::-1], interpolation=cv2.INTER_NEAREST)

                tokens[0] = cv2.resize(tokens[0], img.shape[-2::-1], interpolation=cv2.INTER_NEAREST)
                tokens[1] = cv2.resize(tokens[1], img.shape[-2::-1], interpolation=cv2.INTER_NEAREST)
                tokens[2] = cv2.resize(tokens[2], img.shape[-2::-1], interpolation=cv2.INTER_NEAREST)
                # tokens[3] = cv2.resize(tokens[3], img.shape[-2::-1], interpolation=cv2.INTER_NEAREST)

                horizontal_space_size = 60
                vertical_space_size = 60
                height, width, channels = img.shape
                horizontal_space = 255 * np.ones((height, horizontal_space_size, channels), dtype=img.dtype)
                vertical_space = 255 * np.ones((vertical_space_size, 4 * width + 3 * horizontal_space_size, channels), dtype=img.dtype)
                vertical_space = 255 * np.ones((vertical_space_size, 4 * width, channels), dtype=img.dtype)
            prev_states = states

            if collect_predictions:
                current_labels, valid_batch_indices = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
                # Store backbone features that correspond to the available labels.
                if len(current_labels) > 0:
                    backbone_feature_selector.add_backbone_features(backbone_features=backbone_features,
                                                                    selected_indices=valid_batch_indices)

                    obj_labels.extend(current_labels)
                    ev_repr_selector.add_event_representations(event_representations=ev_tensors,
                                                               selected_indices=valid_batch_indices)
                    if img is not None and tokens_and_scores[0][0] is not None:
                        predictions, _ = self.mdl.forward_detect(backbone_features=backbone_features)
                        pred_processed = postprocess(prediction=predictions,
                                                    num_classes=self.mdl_config.head.num_classes,
                                                    conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                                    nms_thre=self.mdl_config.postprocess.nms_threshold)
                        loaded_labels_proph, yolox_preds_proph = to_prophesee(copy.deepcopy(current_labels), pred_processed)
                        label_img, pred_img = np.ascontiguousarray(img), np.ascontiguousarray(img.copy())
                        draw_bboxes(label_img, loaded_labels_proph[0], labelmap=LABELMAP_GEN4_SHORT, put_score=False)
                        draw_bboxes(pred_img, yolox_preds_proph[0], labelmap=LABELMAP_GEN4_SHORT)
                        # pred_img_2x = cv2.resize(pred_img, [pred_img.shape[1] * 2, pred_img.shape[0] * 2], interpolation=cv2.INTER_NEAREST)

                        # row1 = np.concatenate([label_img, horizontal_space, scores[0], 
                        #                        horizontal_space, scores[1], 
                        #                        horizontal_space, scores[2]], axis=1)
                        # row2 = np.concatenate([pred_img, horizontal_space, tokens[0], 
                        #                        horizontal_space, tokens[1], 
                        #                        horizontal_space, tokens[2]], axis=1)
                        # rows = np.concatenate([row1, vertical_space, row2], axis=0)
                        row1 = np.concatenate([label_img, scores[0], 
                                            scores[1], 
                                            scores[2]], axis=1)
                        row2 = np.concatenate([pred_img, tokens[0], 
                                            tokens[1], 
                                            tokens[2]], axis=1)
                        rows = np.concatenate([row1, vertical_space, row2], axis=0)
                        self.img_tokens.append(rows)
                        self.plt_num += 1
                        # if self.plt_num % 10 == 0:
                            
                        #     height, width = rows.shape[:2]
                        #     plt.figure(figsize=(width / 100, height / 100), dpi=100)
                        #     plt.imshow(rows[:, :, ::-1], aspect='auto')
                        #     plt.axis('off')
                        #     plt.savefig('rows_visualization.png', bbox_inches='tight', pad_inches=0)
                            
                elif img is not None and tokens_and_scores[0][0] is not None:
                    label_img, pred_img = np.ascontiguousarray(img), np.ascontiguousarray(img.copy())
                    pred_img_2x = cv2.resize(pred_img, [pred_img.shape[1] * 2, pred_img.shape[0] * 2], interpolation=cv2.INTER_NEAREST)
                    row1 = np.concatenate([label_img, scores[0], 
                                            scores[1], 
                                            scores[2]], axis=1)
                    row2 = np.concatenate([pred_img, tokens[0], 
                                            tokens[1], 
                                            tokens[2]], axis=1)
                    rows = np.concatenate([row1, vertical_space, row2], axis=0)
                    self.img_tokens.append(rows) 
        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)
        # return {ObjDetOutput.SKIP_VIZ: True}
        # # 将数据添加到 DataFrame 中
        # ratios_series = pd.Series(ratios.mean().item())
        # index_lens = [x for x in index_lens]
        # self.dataframe = pd.concat([self.dataframe, pd.DataFrame({'ratios': ratios_series, 
        #                                       'index_lens_stage_1': pd.Series(index_lens[0]), 
        #                                       'index_lens_stage_2': pd.Series(index_lens[1]), 
        #                                       'index_lens_stage_3': pd.Series(index_lens[2]), 
        #                                       'index_lens_stage_4': pd.Series(index_lens[3])})])

        
        # return {ObjDetOutput.SKIP_VIZ: True}
        if len(obj_labels) == 0:
            return {ObjDetOutput.SKIP_VIZ: True}
        selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()
        predictions, _ = self.mdl.forward_detect(backbone_features=selected_backbone_features)

        pred_processed = postprocess(prediction=predictions,
                                     num_classes=self.mdl_config.head.num_classes,
                                     conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                     nms_thre=self.mdl_config.postprocess.nms_threshold)

        loaded_labels_proph, yolox_preds_proph = to_prophesee(obj_labels, pred_processed)

        # For visualization, we only use the last item (per batch).
        output = {
            ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-1],
            ObjDetOutput.PRED_PROPH: yolox_preds_proph[-1],
            ObjDetOutput.EV_REPR: ev_repr_selector.get_event_representations_as_list(start_idx=-1)[0],
            ObjDetOutput.SKIP_VIZ: False
        }

        self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
        self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)
            
        return output

    def smooth_loss(self, step_loss, idx):
        # if self.trainer.global_step == 0:
        #     self.iou_loss, self.conf_loss, self.cls_loss, self.p_loss = 0, 0, 0, 0
        if idx == 1:
            self.stage1 = (self.stage1 * (self.trainer.global_step) + step_loss) / (self.trainer.global_step + 1)
        elif idx == 2:
            self.stage2 = (self.stage2 * (self.trainer.global_step) + step_loss) / (self.trainer.global_step + 1)
        elif idx == 3:
            self.stage3 = (self.stage3 * (self.trainer.global_step) + step_loss) / (self.trainer.global_step + 1)
        elif idx == 4:
            self.stage4 = (self.stage4 * (self.trainer.global_step) + step_loss) / (self.trainer.global_step + 1)
        
    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode=Mode.VAL)

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode=Mode.TEST)

    def run_psee_evaluator(self, mode: Mode, need_return=False):
        psee_evaluator = self.mode_2_psee_evaluator[mode]
        batch_size = self.mode_2_batch_size[mode]
        hw_tuple = self.mode_2_hw[mode]
        AP = 0
        if psee_evaluator is None:
            warn(f'psee_evaluator is None in {mode=}', UserWarning, stacklevel=2)
            return
        assert batch_size is not None
        assert hw_tuple is not None
        if psee_evaluator.has_data():
            metrics = psee_evaluator.evaluate_buffer(img_height=hw_tuple[0],
                                                     img_width=hw_tuple[1])
            assert metrics is not None

            prefix = f'{mode_2_string[mode]}/'
            step = self.trainer.global_step
            log_dict = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    value = torch.tensor(v)
                elif isinstance(v, np.ndarray):
                    value = torch.from_numpy(v)
                elif isinstance(v, torch.Tensor):
                    value = v
                else:
                    raise NotImplementedError
                assert value.ndim == 0, f'tensor must be a scalar.\n{v=}\n{type(v)=}\n{value=}\n{type(value)=}'
                # put them on the current device to avoid this error: https://github.com/Lightning-AI/lightning/discussions/2529
                log_dict[f'{prefix}{k}'] = value.to(self.device)
            # Somehow self.log does not work when we eval during the training epoch.
            self.log_dict(log_dict, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            if dist.is_available() and dist.is_initialized():
                # We now have to manually sync (average the metrics) across processes in case of distributed training.
                # NOTE: This is necessary to ensure that we have the same numbers for the checkpoint metric (metadata)
                # and wandb metric:
                # - checkpoint callback is using the self.log function which uses global sync (avg across ranks)
                # - wandb uses log_metrics that we reduce manually to global rank 0
                dist.barrier()
                for k, v in log_dict.items():
                    dist.reduce(log_dict[k], dst=0, op=dist.ReduceOp.SUM)
                    if dist.get_rank() == 0:
                        log_dict[k] /= dist.get_world_size()
            if self.trainer.is_global_zero:
                # For some reason we need to increase the step by 2 to enable consistent logging in wandb here.
                # I might not understand wandb login correctly. This works reasonably well for now.
                add_hack = 2
                self.logger.log_metrics(metrics=log_dict, step=step + add_hack)

                # Determine the maximum length of the keys
                max_key_length = max(len(key) for key in log_dict.keys())
                # Print the table
                print(f"{'Metric':<{max_key_length}} | Value")
                print("-" * (max_key_length + 3) + "+" + "-" * 6)
                for key, value in log_dict.items():
                    if 'AP_S' not in key and 'AP_M' not in key and 'AP_L' not in key:
                        value = f"{value * 100:.4f}%" 
                        print(f"{key:<{max_key_length}} | {value}")
                AP = f"{log_dict['test/AP'] * 100:.4f}%"
            psee_evaluator.reset_buffer()
        else:
            warn(f'psee_evaluator has not data in {mode=}', UserWarning, stacklevel=2)
        if need_return:
            return AP

    def on_train_batch_start(self, batch, batch_idx) -> None:
        # Display learning rate in pbar
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.trainer._logger_connector.progress_bar_metrics['lr'] = lr
        
    def on_train_epoch_end(self) -> None:
        mode = Mode.TRAIN
        if mode in self.mode_2_psee_evaluator and \
                self.train_metrics_config.detection_metrics_every_n_steps is None and \
                self.mode_2_hw[mode] is not None:
            # For some reason PL calls this function when resuming.
            # We don't know yet the value of train_height_width, so we skip this
            self.run_psee_evaluator(mode=mode)

    def on_validation_epoch_end(self) -> None:
        mode = Mode.VAL
        excel_file = './validation_logs/val2.xlsx'
        self.dataframe.to_excel(excel_file, index=False)

        if self.started_training:
            assert self.mode_2_psee_evaluator[mode].has_data()
            self.run_psee_evaluator(mode=mode)

    def on_test_epoch_end(self) -> None:
        mode = Mode.TEST
        excel_file = './validation_logs/test2.xlsx'
        self.dataframe.to_excel(excel_file, index=False)
        assert self.mode_2_psee_evaluator[mode].has_data()
        self.run_psee_evaluator(mode=mode)

    def configure_optimizers(self) -> Any:
        lr = self.train_config.learning_rate
        weight_decay = self.train_config.weight_decay
        optimizer = th.optim.AdamW(self.mdl.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_params = self.train_config.lr_scheduler
        if not scheduler_params.use:
            return optimizer

        total_steps = scheduler_params.total_steps
        assert total_steps is not None
        assert total_steps > 0
        # Here we interpret the final lr as max_lr/final_div_factor.
        # Note that Pytorch OneCycleLR interprets it as initial_lr/final_div_factor:
        final_div_factor_pytorch = scheduler_params.final_div_factor / scheduler_params.div_factor
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            div_factor=scheduler_params.div_factor,
            final_div_factor=final_div_factor_pytorch,
            total_steps=total_steps,
            pct_start=scheduler_params.pct_start,
            cycle_momentum=False,
            anneal_strategy='linear')
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
