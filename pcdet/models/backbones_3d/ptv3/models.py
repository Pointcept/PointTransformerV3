"""
Point Transformer V3 - v6m1

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
from functools import partial
from addict import Dict
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath

from .structure import Point as Point_
from .serialization import encode
from .modules import PointModule, PointSequential


class Point(Point_):
    def serialization(self, depth=None, order="z"):
        """
        Point Cloud Serialization
        """
        assert "grid_coord" in self.keys()
        if isinstance(order, str):
            order = [order]

        # add 1 to make grid space support shift order
        if depth is None:
            depth = int(self.grid_coord.max() + 1).bit_length()
        self["depth"] = depth
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        code = [
            encode(self.grid_coord, self.batch, self.depth, order=order_)
            for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )
        self["code"] = code
        self["order"] = order
        self["inverse"] = inverse


class PDNorm(PointModule):
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        if isinstance(point.condition, str):
            condition = point.condition
        else:
            condition = point.condition[0]
        if self.decouple:
            assert condition in self.conditions
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.norm
        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        shift=False,
        order_index=0,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.shift = shift
        self.order_index = order_index

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads)

    def patch_padding(self, data, fill_value=0):
        pad = data.new_full(
            (self.patch_size - data.shape[0] % self.patch_size,) + data.shape[1:],
            fill_value,
        )
        return torch.cat([data, pad], dim=0)

    @torch.no_grad()
    def get_attn_mask(self, point):
        K = self.patch_size
        mask_key = "mask"
        if self.shift:
            mask_key = mask_key + "_shift"
        if mask_key not in point.keys():
            batch = self.patch_padding(point.batch, len(point.offset))
            if self.shift:
                batch[: K // 2] = len(point.offset) + 1
                batch = batch.roll(-K // 2, 0)
            batch = batch.reshape(-1, K)
            mask = batch.unsqueeze(2) - batch.unsqueeze(1)
            mask = mask.masked_fill(mask != 0, -1e3)
            point[mask_key] = mask
        return point[mask_key]

    @torch.no_grad()
    def get_rel_pos(self, point):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if self.shift:
            rel_pos_key = rel_pos_key + "_shift"
        if rel_pos_key not in point.keys():
            grid_coord = self.patch_padding(
                point.grid_coord[point.order[self.order_index]]
            )
            if self.shift:
                grid_coord = grid_coord.roll(-K // 2, 0)
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    def forward(self, point):
        H = self.num_heads
        K = self.patch_size
        C = self.channels

        order = point.order[self.order_index]
        inverse = point.inverse[self.order_index]

        # padding and reshape feat and batch for serialized point patch
        feat = self.patch_padding(point.feat[order])
        if self.shift:
            feat = feat.roll(-K // 2, 0)
        feat = feat.reshape(-1, K, C)

        # attention mask & relative position
        mask = self.get_attn_mask(point)
        rel_pos = self.get_rel_pos(point)

        # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
        qkv = self.qkv(feat).reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)  # (N', H, K, C')

        # attn
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
        attn = attn + self.rpe(rel_pos) + mask.unsqueeze(1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        feat = (attn @ v).transpose(1, 2).reshape(-1, C)  # (N', K, H, C') -> (N, C)
        if self.shift:
            feat = feat.roll(K // 2, 0)
        feat = feat[inverse]

        # ffn
        feat = feat[: point.feat.shape[0]]
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        shift=False,
        order_index=0,
        cpe_indice_key=None,
        sp_dim=3
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            getattr(spconv, f"SubMConv{sp_dim}d")(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            shift=shift,
            order_index=order_index,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=1,
        norm_layer=None,
        act_layer=None,
        reduce="max",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.reduce = reduce
        assert reduce in ["sum", "mean", "min", "max"]

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        assert (
            point.depth > self.depth
        ), f"Current point with depth {point.depth} can not support SerializedPooling with depth {self.depth}"
        assert {"code", "order", "inverse", "depth"}.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.code >> self.depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            grid_coord=point.grid_coord[head_indices] >> self.depth,
            code=code,
            order=order,
            inverse=inverse,
            depth=point.depth - self.depth,
            batch=point.batch[head_indices],
            sparse_shape=[i >> self.depth for i in point.sparse_shape]
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        # hook parent
        point["cluster"] = cluster
        point_dict["parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

    def forward(self, point):
        assert "parent" in point.keys()
        parent = point.pop("parent")
        cluster = parent.pop("cluster")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[cluster]
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
        sp_dim=3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        self.stem = PointSequential(
            conv=getattr(spconv, f"SubMConv{sp_dim}d")(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


class PointTransformerV3(torch.nn.Module):
    def __init__(
        self,
        model_cfg,
        input_channels,
        grid_size,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        order="z",
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("NuScenes", "SemanticKITTI", "Waymo"),
        **kwargs
    ):
        super().__init__()
        enc_depths = model_cfg.enc_depths
        enc_channels = model_cfg.enc_channels
        enc_num_head = model_cfg.enc_num_head
        enc_patch_size = model_cfg.enc_patch_size
        self.sp_choice_idx = model_cfg.sp_choice_idx  # [1, 2] for pillar
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.sparse_shape = grid_size[::-1]
        self.num_point_features = enc_channels[-1]

        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=input_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
            sp_dim=len(self.sp_choice_idx)
        )

        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]

        self.enc = PointSequential()

        # encoder
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        shift=False if i % 2 == 0 else True,
                        order_index=(i >> 1) % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        sp_dim=len(self.sp_choice_idx)
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # self.out = spconv.SparseSequential(
        #     # [200, 150, 5] -> [200, 150, 2]
        #     spconv.SparseConv3d(
        #         enc_channels[-1], enc_channels[-1], (3, 1, 1),
        #         stride=(2, 1, 1), padding=0,
        #         bias=False, indice_key='spconv_down2'),
        #     bn_layer(128),
        #     nn.ReLU(),
        # )

    def forward(self, batch_dict):
        feat, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch = voxel_coords[:, 0].long()
        grid_coord = voxel_coords[:, 1:].long()
        point = Point(feat=feat, batch=batch, grid_coord=grid_coord, sparse_shape=self.sparse_shape)
        point.serialization(order=self.order)
        point.sparsify(sp_choice_idx=self.sp_choice_idx)

        point = self.embedding(point)
        point = self.enc(point)

        # batch_dict.update({
        #     'encoded_spconv_tensor': self.out(point.sparse_conv_feat),
        #     'encoded_spconv_tensor_stride': 2 ** (self.num_stages - 1)
        # })
        batch_dict['pillar_features'] = batch_dict['voxel_features'] = point.feat
        batch_dict['voxel_coords'] = voxel_coords
        return batch_dict
