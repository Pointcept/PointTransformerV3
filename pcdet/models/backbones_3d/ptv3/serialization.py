import torch
import ocnn
try:
    import hilbert_torch
except:
    hilbert_torch=None


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        code = grid_coord_to_z_order_code(grid_coord, depth=depth)
    elif order == "z-trans":
        code = grid_coord_to_z_order_code(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = grid_coord_to_hilbert_code(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = grid_coord_to_hilbert_code(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


@torch.inference_mode()
def decode(code, depth=16, order="z"):
    assert order in {"z", "hilbert"}
    batch = code >> depth * 3
    code = code & ((1 << depth * 3) - 1)
    if order == "z":
        grid_coord = z_order_code_to_grid_coord(code, depth=depth)
    elif order == "hilbert":
        grid_coord = hilbert_code_to_grid_coord(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch


def grid_coord_to_z_order_code(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = ocnn.octree.xyz2key(x, y, z, b=None, depth=depth)
    return code


def z_order_code_to_grid_coord(code: torch.Tensor, depth):
    x, y, z, _ = ocnn.octree.key2xyz(code, depth=depth)
    grid_coord = torch.stack([x, y, z], dim=-1)  # (N,  3)
    return grid_coord


def grid_coord_to_hilbert_code(grid_coord: torch.Tensor, depth: int = 16):
    return hilbert_torch.encode(grid_coord, num_dims=3, num_bits=depth)


def hilbert_code_to_grid_coord(code: torch.Tensor, depth: int = 16):
    return hilbert_torch.decode(code, num_dims=3, num_bits=depth)
