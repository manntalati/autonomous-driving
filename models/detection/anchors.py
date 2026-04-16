import torch
from typing import Dict, List, Optional, Tuple


class AnchorGenerator:
    def __init__(self, scales: List[float], aspect_ratios: List[float], strides: List[int]) -> None:
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        # Cache: (feature_map_sizes, image_size, device) → anchors tensor.
        # Feature-map sizes are fixed during training, so this collapses
        # anchor generation from ~O(H*W*A) Python iterations to a dict lookup.
        self._cache: Dict[Tuple, torch.Tensor] = {}

    def generate_for_level(self, feature_h: int, feature_w: int, stride: int, scale: float) -> torch.Tensor:
        """
        Tile anchors across a single feature map level — vectorized.
        Ordering (preserved from original impl): ratio → row → col, so the
        k-th anchor corresponds to (a = k // (H*W), h = (k // W) % H, w = k % W).
        Returns: (H*W*num_ratios, 4) float32 in [x1,y1,x2,y2].
        """
        ratios = torch.tensor(self.aspect_ratios, dtype=torch.float32)
        w = scale * torch.sqrt(ratios)          # (A,)
        h = scale / torch.sqrt(ratios)          # (A,)

        shifts_x = (torch.arange(feature_w, dtype=torch.float32) + 0.5) * stride   # (W,)
        shifts_y = (torch.arange(feature_h, dtype=torch.float32) + 0.5) * stride   # (H,)
        cy_grid, cx_grid = torch.meshgrid(shifts_y, shifts_x, indexing="ij")       # (H, W)

        # Broadcast to (A, H, W). Order matters: stacking along dim 0 (ratios) first
        # gives reshape(-1, 4) iteration order: ratio → row → col.
        A = ratios.shape[0]
        cx = cx_grid[None, :, :].expand(A, feature_h, feature_w)
        cy = cy_grid[None, :, :].expand(A, feature_h, feature_w)
        half_w = (w * 0.5)[:, None, None].expand(A, feature_h, feature_w)
        half_h = (h * 0.5)[:, None, None].expand(A, feature_h, feature_w)

        x1 = cx - half_w
        y1 = cy - half_h
        x2 = cx + half_w
        y2 = cy + half_h
        anchors = torch.stack([x1, y1, x2, y2], dim=-1)   # (A, H, W, 4)
        return anchors.reshape(-1, 4).contiguous()

    def generate_all(self, feature_map_sizes: List[Tuple[int, int]], image_size: Tuple[int, int], device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate anchors for all FPN levels, concatenate, clamp to image bounds.
        Results are cached per (sizes, image_size, device) so repeat calls cost O(1).
        Args: feature_map_sizes — list of (H, W) per level; image_size — (H, W) of the image;
              device — optional torch device to place the result on.
        Returns: (total_anchors, 4) float32 tensor.
        """
        sizes_key = tuple((int(h), int(w)) for h, w in feature_map_sizes)
        img_key = (int(image_size[0]), int(image_size[1]))
        dev_key = str(device) if device is not None else "cpu"
        key = (sizes_key, img_key, dev_key)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        anchors_list = [
            self.generate_for_level(h, w, self.strides[i], self.scales[i])
            for i, (h, w) in enumerate(feature_map_sizes)
        ]
        anchors = torch.cat(anchors_list, dim=0)
        img_h, img_w = image_size
        anchors[:, 0].clamp_(min=0, max=img_w)
        anchors[:, 1].clamp_(min=0, max=img_h)
        anchors[:, 2].clamp_(min=0, max=img_w)
        anchors[:, 3].clamp_(min=0, max=img_h)
        if device is not None:
            anchors = anchors.to(device)
        self._cache[key] = anchors
        return anchors
