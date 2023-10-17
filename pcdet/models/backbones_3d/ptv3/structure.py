import torch
import spconv.pytorch as spconv
import ocnn
from addict import Dict
from .serialization import encode, decode
from .utils import offset2batch, batch2offset


class Point(Dict):
    """
    Point Structure of Pointcept

    A minimum definition of a Point (point cloud) in Pointcept contains:
    - "coord": original coordinate of point cloud;
    or
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    (related to Serialization)
    - "depth": depth of serialization, 2 ** depth * grid_size describe the maximum of point cloud range;
    - "z_order_code": code describe batched point order with a z-order curve;
    - "z_order": index sequence of the point cloud, sorted by z_order_code;
    - "hilbert_order_code": code describe batched point order with a hilbert curve;
    - "hilbert_order": index sequence of the point cloud, sorted by hilbert_order_code;
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert "coord" in self.keys() or "grid_coord" in self.keys()
        # # if both "offset" and "batch" do not exist, assume bs is 1
        # if "offset" not in self.keys() and "batch" not in self.keys():
        #     if "coord" in self.keys():
        #         self["offset"] = torch.tensor(
        #             [len(self.coord)],
        #             device=self.coord.device,
        #             dtype=torch.long,
        #         )
        #     elif "grid_coord" in self.keys():
        #         self["offset"] = torch.tensor(
        #             [len(self.grid_coord)],
        #             device=self.grid_coord.device,
        #             dtype=torch.long,
        #         )

        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        if "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, depth=None):
        """
        Point Cloud Serialization
        """
        assert "grid_coord" in self.keys()
        # add 1 to make grid space support shift order
        if depth is None:
            depth = int(self.grid_coord.max() + 1).bit_length()
        self["depth"] = depth
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        self["z_code"] = encode(self.grid_coord, self.batch, self.depth, order="z")
        order = torch.argsort(self["z_code"])
        inverse = torch.zeros_like(order)
        inverse[order] = torch.arange(0, len(order), device=order.device)
        self["z_order"] = order
        self["z_inverse"] = inverse

    def sparsify(self, pad=96):
        """
        Point Cloud Serialization

        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.

        pad: padding sparse for sparse shape.
        """
        assert {"grid_coord", "feat", "batch"}.issubset(self.keys())
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(torch.max(self.grid_coord, dim=0).values, pad).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat

    def octreetization(self, depth=None, full_depth=None):
        """
        Point Cloud Octreelization

        Generate octree with OCNN
        """
        assert {"grid_coord", "feat", "batch"}.issubset(self.keys())
        # add 1 to make grid space support shift order
        if depth is None:
            if "depth" in self.keys():
                depth = self.depth
            else:
                depth = int(self.grid_coord.max() + 1).bit_length()
        if full_depth is None:
            full_depth = 2
        self["depth"] = depth
        assert depth <= 16  # maximum in ocnn

        # [0, 2**depth] -> [0, 2] -> [-1, 1]
        coord = self.grid_coord / 2 ** (self.depth - 1) - 1.0
        point = ocnn.octree.Points(
            points=coord,
            features=self.feat,
            batch_id=self.batch.unsqueeze(-1),
            batch_size=self.batch[-1] + 1,
        )
        octree = ocnn.octree.Octree(
            depth=depth,
            full_depth=full_depth,
            batch_size=self.batch[-1] + 1,
            device=coord.device,
        )
        octree.build_octree(point)
        octree.construct_all_neigh()
        self["octree"] = octree
