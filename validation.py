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
from modules.utils.fetch import fetch_data_module, fetch_model_module
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
    # Data
    # ---------------------
    data_module = fetch_data_module(config=config)

    # ---------------------
    # Logging and Checkpoints
    # ---------------------
    logger = CSVLogger(save_dir='./validation_logs')
    ckpt_path = Path(config.checkpoint)

    # ---------------------
    # Model
    # ---------------------
    
    module = fetch_model_module(config=config)
    module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config}, strict=False)

    # A-FLOPs # 256, 320
    model = fetch_model_module(config=config)
    model = model.load_from_checkpoint(str(ckpt_path), **{'full_config': config}, strict=False).cuda()
    dataset = fetch_data_module(config=config)
    dataset.setup(stage='test')
    dataset = dataset.test_dataset
    # model = torch.compile(model)
    sparsity = 1 - 0.005
    if utils.is_main_process() and config.benchmark:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        fps = compute_fps(model, dataset, num_iters=100, batch_size=1, sparsity=sparsity)
        b32fps = compute_fps(model, dataset, num_iters=10, batch_size=4, sparsity=sparsity)
        if config.benchmark_only:
            gflops = compute_gflops(model, dataset, approximated=False, sparsity=sparsity)
        else:
            gflops = compute_gflops(model, dataset, approximated=True, sparsity=sparsity)

        # gflops = '???'
        #model = torch.jit.script(model)
        # model = torch.jit.trace(model, images, strict=False)
        tab_keys = ["#Params(M)", "GFLOPs", "FPS", "B4FPS"]
        tab_vals = [n_params / 10 ** 6, gflops, fps, b32fps]
        table = tabulate([tab_vals], headers=tab_keys, tablefmt="pipe",
                        floatfmt=".3f", stralign="center", numalign="center")
        print("===== Benchmark (Crude Approx.) =====\n" + table)
    model = None

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = [ModelSummary(max_depth=2)]

    # ---------------------
    # Validation
    # ---------------------

    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        default_root_dir=None,
        devices=gpus,
        logger=logger,
        log_every_n_steps=100,
        precision=config.training.precision,
        move_metrics_to_cpu=False,
    )
    with torch.inference_mode():
        if config.use_test_set:
            trainer.test(model=module, datamodule=data_module, ckpt_path=str(ckpt_path))
        else:
            trainer.validate(model=module, datamodule=data_module, ckpt_path=str(ckpt_path))


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
