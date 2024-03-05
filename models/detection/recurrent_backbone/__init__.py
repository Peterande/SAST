from omegaconf import DictConfig

from .sast_rnn import RNNDetector as SASTRNNDetector


def build_recurrent_backbone(backbone_cfg: DictConfig):
    name = backbone_cfg.name
    if name == 'SASTRNN':
        return SASTRNNDetector(backbone_cfg)
    else:
        raise NotImplementedError
