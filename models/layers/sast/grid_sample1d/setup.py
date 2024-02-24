from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='grid_sample1d_cuda',
    ext_modules=[
        CUDAExtension('grid_sample1d_cuda', [
            'grid_sample1d_cuda.cpp',
            'grid_sample1d_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    })