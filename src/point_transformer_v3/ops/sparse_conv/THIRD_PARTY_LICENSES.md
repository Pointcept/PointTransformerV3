# Third-party licenses

The code of point-transformer-v3 is vendored from Pointcept2 (Pointcept's own code; license in
LICENSE and NOTICE at the repository root). Parts of it are adapted from third-party projects
under the MIT license; their notices follow. Nothing in this package is copied from the Sonata
repository (facebookresearch/sonata); Sonata's weights are downloaded at runtime and are licensed
separately (CC BY-NC 4.0, see NOTICE).

## Where the third-party code is

- `point_transformer_v3/ops/sparse_conv/_triton_kernels.py` and `_triton_ops.py` (used by
  `triton_impl.py`): the Triton implicit-GEMM submanifold conv kernels, adapted from
  ocnn-pytorch, whose kernels are in turn adapted from FlexGEMM.
- `point_transformer_v3/ops/neighbors.py:kmap_from_coarse` and `_neighbors_triton.py`: port of
  ocnn-pytorch's top-down neighbor construction (`Octree.construct_neigh_by_parent`,
  `Octree._get_lut`).

## Sources

- ocnn-pytorch: https://github.com/octree-nn/ocnn-pytorch, commit
  `7521c22e2921a0bd8e9285044c842ff6fa2042e0` (files `ocnn/nn/kernels/*.py`,
  `ocnn/nn/octree_conv_t.py`, `ocnn/octree/octree.py`).
- FlexGEMM: https://github.com/JeffreyXiang/FlexGEMM (via `ocnn/nn/kernels/`, whose `LICENSE` is
  reproduced below).

## ocnn-pytorch

```
MIT License

Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## FlexGEMM (as shipped in ocnn-pytorch `ocnn/nn/kernels/LICENSE`)

```
MIT License

Copyright (c) 2025 Jianfeng Xiang (belljig@outlook.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
