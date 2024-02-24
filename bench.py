import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pathlib import Path

import torch
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelSummary

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_data_module, fetch_model_module, fetch_backbone_module
import utils.misc as utils
from tabulate import tabulate
from utils.benchmark import compute_fps, compute_gflops

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # ---------------------
    # GPU options
    # ---------------------
    gpus = config.hardware.gpus
    assert isinstance(gpus, int), 'no more than 1 GPU supported'
    gpus = [gpus]

    # ---------------------
    # Checkpoints
    # ---------------------
    ckpt_path = Path(config.checkpoint)

    # ---------------------
    # Model
    # ---------------------
    model = fetch_backbone_module(config=config).cuda()
    model = model.load_from_checkpoint(str(ckpt_path), **{'full_config': config}, strict=False)

    # ---------------------
    # Data
    # ---------------------
    dataset = fetch_data_module(config=config)
    dataset.setup(stage='test')
    dataset = dataset.test_dataset


    if utils.is_main_process() and config.benchmark:
        sparsity = 1 - 1e-3
        batch_size = 1
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        bfps = compute_fps(model, dataset, num_iters=10, batch_size=batch_size, sparsity=sparsity)
        fps = compute_fps(model, dataset, num_iters=100, batch_size=1, sparsity=sparsity)
        
        if config.benchmark_only:
            gflops = compute_gflops(model, dataset, approximated=False, sparsity=sparsity)
        else:
            gflops = compute_gflops(model, dataset, approximated=True, sparsity=sparsity)

        tab_keys = ["#Params(M)", "GFLOPs", "FPS", "B" + str(batch_size) + "FPS"]
        tab_vals = [n_params / 10 ** 6, gflops, fps, bfps]
        table = tabulate([tab_vals], headers=tab_keys, tablefmt="pipe",
                        floatfmt=".3f", stralign="center", numalign="center")
        print("===== Benchmark (Crude Approx.) =====\n" + table)
        

if __name__ == '__main__':
    main()
