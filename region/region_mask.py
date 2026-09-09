#!/usr/bin/env python3
"""World-frame dynamic-region mask.

The step2 region-aware pass needs a fast point-in-dynamic-region query for
many observations.  This module rasterizes the high-speed dynamic polygons to
a metric grid and exposes ``contains_point`` / ``contains_points``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Bounds = Tuple[float, float, float, float]


@dataclass
class DynamicRegionMask:
    bounds: Bounds
    resolution: float
    mask: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.mask.shape[0]), int(self.mask.shape[1])

    @classmethod
    def empty(cls, resolution: float = 1.0) -> "DynamicRegionMask":
        return cls(bounds=(0.0, 0.0, 1.0, 1.0), resolution=float(resolution),
                   mask=np.zeros((2, 2), dtype=np.uint8))

    @classmethod
    def from_polygons(
            cls,
            polygons: Sequence[Sequence[Sequence[float]]],
            *,
            resolution: float = 1.0,
            margin: float = 5.0,
    ) -> "DynamicRegionMask":
        valid = [np.asarray(polygon, dtype=np.float64)
                 for polygon in polygons if len(polygon) >= 3]
        if not valid:
            return cls.empty(resolution)
        all_points = np.vstack(valid)
        xmin, ymin = all_points.min(axis=0) - float(margin)
        xmax, ymax = all_points.max(axis=0) + float(margin)
        width = int(math.ceil((xmax - xmin) / resolution)) + 1
        height = int(math.ceil((ymax - ymin) / resolution)) + 1
        mask = np.zeros((max(height, 1), max(width, 1)), dtype=np.uint8)
        for polygon in valid:
            columns = np.clip(
                ((polygon[:, 0] - xmin) / resolution).astype(np.int64),
                0, mask.shape[1] - 1)
            rows = np.clip(
                ((polygon[:, 1] - ymin) / resolution).astype(np.int64),
                0, mask.shape[0] - 1)
            cv2.fillPoly(mask, [np.column_stack([columns, rows]).astype(np.int32)], 1)
        return cls(bounds=(float(xmin), float(ymin), float(xmax), float(ymax)),
                   resolution=float(resolution), mask=mask)

    @classmethod
    def from_regions_json(cls, path: Path,
                          *, resolution: float = 1.0,
                          margin: float = 5.0) -> "DynamicRegionMask":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        polygons = [item["polygon"]
                    for item in data.get("dynamic_polygons", [])]
        return cls.from_polygons(polygons, resolution=resolution,
                                 margin=margin)

    def contains_points(self, points: Any) -> np.ndarray:
        array = np.asarray(points, dtype=np.float64)
        if array.size == 0:
            return np.zeros(0, dtype=bool)
        array = array.reshape(-1, 2)
        columns = np.floor(
            (array[:, 0] - self.bounds[0]) / self.resolution).astype(np.int64)
        rows = np.floor(
            (array[:, 1] - self.bounds[1]) / self.resolution).astype(np.int64)
        inside_bounds = (
            (columns >= 0) & (columns < self.mask.shape[1])
            & (rows >= 0) & (rows < self.mask.shape[0]))
        result = np.zeros(len(array), dtype=bool)
        if np.any(inside_bounds):
            result[inside_bounds] = self.mask[
                rows[inside_bounds], columns[inside_bounds]].astype(bool)
        return result

    def contains_point(self, x: float, y: float) -> bool:
        return bool(self.contains_points(np.asarray([[x, y]], dtype=np.float64))[0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bounds": {
                "xmin": round(float(self.bounds[0]), 4),
                "ymin": round(float(self.bounds[1]), 4),
                "xmax": round(float(self.bounds[2]), 4),
                "ymax": round(float(self.bounds[3]), 4),
            },
            "resolution": float(self.resolution),
            "dynamic_cells": int(np.count_nonzero(self.mask)),
            "dynamic_area_m2": round(
                float(np.count_nonzero(self.mask)) * self.resolution ** 2, 3),
        }


def build_mask_from_regions_json(path: Path,
                                 resolution: float = 1.0) -> DynamicRegionMask:
    return DynamicRegionMask.from_regions_json(path, resolution=resolution)
