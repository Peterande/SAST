from collections import defaultdict
import time
from typing import Any, Counter, DefaultDict, Tuple, Dict, Optional
import warnings

import numpy as np
import torch
from torch import nn
import tqdm

from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import Handle
from data.utils.types import DataType
from utils.padding import InputPadderFromShape
import pandas as pd


@torch.no_grad()
def measure_average_inference_time_iter(model, inputs, num_iters=100, warm_iters=5):
    ts = []
    # note that warm-up iters. are excluded from the total iters.
    for iter_ in tqdm.tqdm(range(warm_iters + num_iters)):
        torch.cuda.synchronize()
        t_ = time.perf_counter()
        model(inputs)
        torch.cuda.synchronize()
        t = time.perf_counter() - t_
        if iter_ >= warm_iters:
          ts.append(t)
    return sum(ts) / len(ts)


@torch.no_grad()
def measure_average_inference_time(model, inputs, num_iters=100, warm_iters=10):
    for iter_ in tqdm.tqdm(range(warm_iters + num_iters)):
        if iter_ == warm_iters:
            torch.cuda.synchronize()
            t_ = time.perf_counter()
        model(inputs)
    torch.cuda.synchronize()
    t = time.perf_counter() - t_
    return t / num_iters

def python_ops_mode_for_deform_attn(model, ops_mode):
    def change_ops_mode(module):
        if hasattr(module, "python_ops_for_test"):
            module.python_ops_for_test = ops_mode
    model.apply(change_ops_mode)
    
    
@torch.no_grad()
def compute_fps(model, dataset, num_iters=300, warm_iters=50, batch_size=4, sparsity=0.0):
    print(f"computing fps.. (num_iters={num_iters}, batch_size={batch_size}) "
          f"warm_iters={warm_iters}, batch_size={batch_size}]")
    assert num_iters > 0 and warm_iters >= 0 and batch_size > 0
    model.cuda()
    model.eval()
    inputs = torch.rand((batch_size, 20, 384, 640)).cuda()
    inputs[:] = (inputs[:] > sparsity)
    t = measure_average_inference_time(model, inputs.int(), num_iters, warm_iters)
    # t = measure_average_inference_time(model, dataset, batch_size, num_iters, warm_iters)
    model.train()
    print(f"FPS: {1.0 / t * batch_size}")  
    return 1.0 / t * batch_size
      

@torch.no_grad()
def compute_gflops(model, dataset, approximated=True, sparsity=0.0):
    print(f"computing flops.. (approximated={approximated})")
    model.eval()
    python_ops_mode_for_deform_attn(model, True)
    gflops_list = []
    imsize_list = []
    if approximated:
        # use just a single image to approximate the full compuation
        # the size of the image was found heuristically
        images = [torch.rand((1, 20, 384, 640)).cuda().float()]
        images[0] = (images[0] > sparsity).float()

        for img in tqdm.tqdm(images):
            inputs = img.cuda()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                res = flop_count_without_warnings(model, (inputs,), )[0]
            gflops = sum(res.values()) - res['conv']
            gflops_list.append(gflops)
            imsize_list.append(list(img.shape))
    else:
        # full computation: get the first 100 images of COCO val2017
        input_padder = InputPadderFromShape(desired_hw=(384, 640))
        r_list = []
        for i in tqdm.tqdm(range(len(dataset.datapipe_list))):
            kk = len(dataset.datapipe_list[i])
            for j in tqdm.tqdm(range(kk)):
                inputs = input_padder.pad_tensor_ev_repr(dataset.datapipe_list[i][j][DataType.EV_REPR][-1].unsqueeze(0).float()).cuda()
                r = non_zero_ratio(inputs)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    res = flop_count_without_warnings(model.cuda(), (inputs,), )[0]
                gflops = sum(res.values()) - res['conv']
                gflops_list.append(gflops)
                imsize_list.append(list(inputs.shape))
                r_list.append((r.item(), gflops))
            df = pd.DataFrame(r_list, columns=['r', 'gflops'])
            excel_file_path = 'gflops2.xlsx'
            df.to_excel(excel_file_path, index=False, engine='openpyxl')
    
    if approximated:
        print(f"The image size used for approximation: [1, 20, 384, 640]")
    else:
        print("Average image size of first 100 image of COCO val2017 : "
              f"{np.array(imsize_list).mean(0)}")
        
    print(f"GFLOPs : {np.array(gflops_list).mean()}")
    model.train()
    python_ops_mode_for_deform_attn(model, False)
    return np.array(gflops_list).mean()


def flop_count_without_warnings(
    
    model: nn.Module,
    inputs: Tuple[Any, ...],
    supported_ops: Optional[Dict[str, Handle]] = None,
) -> Tuple[DefaultDict[str, float], Counter[str]]:
    """copied and modified from fvcore.nn.flop_count.py
    
    Given a model and an input to the model, compute the per-operator Gflops
    of the given model.
    Args:
        model (nn.Module): The model to compute flop counts.
        inputs (tuple): Inputs that are passed to `model` to count flops.
            Inputs need to be in a tuple.
        supported_ops (dict(str,Callable) or None) : provide additional
            handlers for extra ops, or overwrite the existing handlers for
            convolution and matmul and einsum. The key is operator name and the value
            is a function that takes (inputs, outputs) of the op. We count
            one Multiply-Add as one FLOP.
    Returns:
        tuple[defaultdict, Counter]: A dictionary that records the number of
            gflops for each operation and a Counter that records the number of
            unsupported operations.
    """
    if supported_ops is None:
        supported_ops = {}
    flop_counter = FlopCountAnalysis(model, inputs).set_op_handle(**supported_ops)
    flop_counter.unsupported_ops_warnings(False)
    flop_counter.uncalled_modules_warnings(False)
    flop_counter.tracer_warnings("no_tracer_warning")
    giga_flops = defaultdict(float)
    for op, flop in flop_counter.by_operator().items():
        giga_flops[op] = flop / 1e9
    return giga_flops, flop_counter.unsupported_ops() 


@torch.no_grad()
def non_zero_ratio(x: torch.Tensor) -> torch.Tensor:
    num_nonzero = torch.sum(torch.sum(x != 0, dtype=torch.int16, dim=[1, 2]), dtype=torch.int32, dim=-1)
    result = x.shape[0] * num_nonzero.float() / x.numel()
    return result