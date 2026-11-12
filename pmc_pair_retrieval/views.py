"""Three-view image preprocessing, view aggregation and vision activation checkpointing.

Every image is resized so that its short side is 224 px (torchvision ``Resize(224)``,
bicubic, antialiased; the long side becomes ``L = int(224 * long / short)``) and cut into
three square 224 x 224 crops along its long axis: start, centre and end. The crop offsets
are ``round(p * (L - 224))`` for the fractions ``p`` in ``VIEW_FRACTIONS`` (0, 0.5, 1);
the centre crop is identical to ``CenterCrop(224)``. For a square image the three crops
coincide. Both configurations use these three views for training and for inference.

Evaluation uses the crops as they are. Training jitters each crop with an isotropic random
resized crop inside the crop, so every view keeps its coverage; the image_to_image
configuration adds colour jitter and a small rotation. Training and evaluation aggregate
the views in the same way: every view embedding is L2-normalised, the three view
embeddings of an image are averaged, and the mean is normalised again.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

# Views per image, for training and inference, and the position of each crop along the long axis.
N_VIEWS = 3
VIEW_FRACTIONS: tuple[float, ...] = (0.0, 0.5, 1.0)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
BICUBIC = T.InterpolationMode.BICUBIC


# --------------------------------------------------------------------------------------
# View geometry
# --------------------------------------------------------------------------------------
def view_offsets(long_side: int, size: int = 224) -> list[int]:
    """Offsets of the three crops along the long axis of a resized image.

    Python's ``round`` (halves to even) is part of the definition: it decides the offset
    of the centre view when ``L - 224`` is odd.
    """
    return [int(round(p * (int(long_side) - int(size)))) for p in VIEW_FRACTIONS]


def view_boxes(width: int, height: int, size: int = 224) -> list[tuple[int, int, int, int]]:
    """Crop boxes ``(left, top, right, bottom)`` of the three views on a resized image of size ``(width, height)``."""
    w, h = int(width), int(height)
    if min(w, h) != int(size):
        raise ValueError(f"view_boxes expects a resized image with short side {size}, got {w}x{h}")
    long_side = max(w, h)
    boxes = []
    for o in view_offsets(long_side, size):
        if w >= h:
            boxes.append((o, 0, o + int(size), int(size)))
        else:
            boxes.append((0, o, int(size), o + int(size)))
    return boxes


def resize_short_side(img: Image.Image, size: int = 224) -> Image.Image:
    """Exactly torchvision ``Resize(size, interpolation=BICUBIC)`` on a PIL image."""
    return TF.resize(img, int(size), interpolation=BICUBIC, antialias=True)


# --------------------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------------------
class MultiViewEvalTransform:
    """PIL image -> tensor ``[3, 3, size, size]`` of the three deterministic view crops."""

    def __init__(self, size: int = 224, mean: Sequence[float] = CLIP_MEAN, std: Sequence[float] = CLIP_STD):
        self.size = int(size)
        self.normalize = T.Normalize(mean, std)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB") if img.mode != "RGB" else img
        img = resize_short_side(img, self.size)
        w, h = img.size
        views = [self.normalize(TF.to_tensor(img.crop(box))) for box in view_boxes(w, h, self.size)]
        return torch.stack(views, dim=0)

    def __repr__(self) -> str:
        return f"MultiViewEvalTransform(views={N_VIEWS}, size={self.size}, fractions={VIEW_FRACTIONS})"


class MultiViewTrainTransform:
    """PIL image -> tensor ``[3, 3, size, size]`` of the three view crops, randomly jittered.

    Each deterministic view crop (see ``MultiViewEvalTransform``) is jittered with
    ``RandomResizedCrop(size, scale=crop_scale, ratio=(1, 1), bicubic)`` applied inside
    the crop. Optional photometric and geometric augmentations follow, per view and in
    this order: ``RandomApply([ColorJitter(brightness, contrast)], p)`` when
    ``color_jitter`` (a mapping with keys ``brightness``, ``contrast`` and ``p``) is
    given, then ``RandomRotation(rotation_degrees)`` when ``rotation_degrees`` is given;
    each view is then converted to a tensor and normalised.

    All random numbers come from the torch generator of the calling process (the
    DataLoader worker), view by view, in the order above. The image_text_to_image
    configuration uses the crop jitter only; image_to_image uses all three.
    """

    def __init__(self, size: int = 224, crop_scale: Sequence[float] = (0.9, 1.0),
                 color_jitter: Mapping[str, float] | None = None, rotation_degrees: float | None = None,
                 mean: Sequence[float] = CLIP_MEAN, std: Sequence[float] = CLIP_STD):
        self.size = int(size)
        self.scale = (float(crop_scale[0]), float(crop_scale[1]))
        self.ratio = (1.0, 1.0)
        post = []
        if color_jitter is not None:
            post.append(T.RandomApply([T.ColorJitter(brightness=color_jitter["brightness"],
                                                     contrast=color_jitter["contrast"])], p=color_jitter["p"]))
        if rotation_degrees is not None:
            post.append(T.RandomRotation(degrees=rotation_degrees))
        post += [T.ToTensor(), T.Normalize(mean, std)]
        self.post = T.Compose(post)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB") if img.mode != "RGB" else img
        img = resize_short_side(img, self.size)
        w, h = img.size
        views = []
        for box in view_boxes(w, h, self.size):
            crop = img.crop(box)
            i, j, hh, ww = T.RandomResizedCrop.get_params(crop, scale=self.scale, ratio=self.ratio)
            crop = TF.resized_crop(crop, i, j, hh, ww, [self.size, self.size], interpolation=BICUBIC, antialias=True)
            views.append(self.post(crop))
        return torch.stack(views, dim=0)

    def __repr__(self) -> str:
        return (f"MultiViewTrainTransform(views={N_VIEWS}, size={self.size}, crop_scale={self.scale}, "
                f"ratio={self.ratio}, post={self.post})")


# --------------------------------------------------------------------------------------
# View aggregation
# --------------------------------------------------------------------------------------
def aggregate_view_embeddings(z: torch.Tensor, n_views: int) -> torch.Tensor:
    """``[N * n_views, D]`` view embeddings -> ``[N, D]`` unit vectors (normalise, mean, renormalise), fp32."""
    z = F.normalize(z.float(), dim=1)
    z = z.view(-1, int(n_views), z.shape[-1]).mean(dim=1)
    return F.normalize(z, dim=1)


def encode_image_views(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``x`` of shape ``[N, 3, 3, H, W]`` (images, views, channels, height, width) -> ``[N, D]`` image embeddings.

    All views of the batch go through the vision tower in one forward pass. Under
    autocast the encoder runs in the autocast dtype; the aggregation is fp32.
    """
    if x.dim() != 5 or int(x.shape[1]) != N_VIEWS:
        raise ValueError(f"expected a batch of shape [N, {N_VIEWS}, 3, H, W], got {tuple(x.shape)}")
    z = model.encode_image(x.flatten(0, 1))
    return aggregate_view_embeddings(z, N_VIEWS)


# --------------------------------------------------------------------------------------
# Activation checkpointing of the vision blocks
# --------------------------------------------------------------------------------------
def _wrap_block_with_checkpoint(block: nn.Module) -> None:
    original_forward = block.forward

    def checkpointed_forward(x, *args, **kwargs):
        # Recompute only when a backward pass can reach this block. Frozen blocks are kept
        # in eval mode during training, so the decision cannot depend on block.training.
        if torch.is_grad_enabled() and (x.requires_grad or any(p.requires_grad for p in block.parameters())):
            return torch.utils.checkpoint.checkpoint(original_forward, x, *args, use_reentrant=False, **kwargs)
        return original_forward(x, *args, **kwargs)

    block.forward = checkpointed_forward


def enable_vision_block_checkpointing(model: nn.Module, mode: str) -> list[int]:
    """Wrap ViT blocks of ``model.visual.trunk`` with non-reentrant activation checkpointing.

    ``mode`` is ``'none'``, ``'trainable'`` (blocks with a trainable parameter) or ``'all'``.
    Checkpointing saves memory only: it preserves the RNG state and does not change the
    computed values. Returns the indices of the wrapped blocks.
    """
    mode = str(mode).lower().strip()
    if mode not in ("none", "trainable", "all"):
        raise ValueError(f"Unknown vision checkpoint mode: {mode}")
    if mode == "none":
        return []
    chosen = []
    for i, blk in enumerate(model.visual.trunk.blocks):
        trainable = any(p.requires_grad for p in blk.parameters())
        if mode == "all" or trainable:
            _wrap_block_with_checkpoint(blk)
            chosen.append(i)
    return chosen
