# -*- coding: utf-8 -*-
"""Площадь пересечения кругов."""
import numpy as np


def circ_iou(a, b):
    (y1, x1, r1), (y2, x2, r2) = a, b
    d = np.hypot(y1 - y2, x1 - x2)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        inter = np.pi * min(r1, r2) ** 2
    else:
        a1 = r1 * r1 * np.arccos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
        a2 = r2 * r2 * np.arccos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
        a3 = 0.5 * np.sqrt(max((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2), 0))
        inter = a1 + a2 - a3
    return inter / (np.pi * r1 * r1 + np.pi * r2 * r2 - inter)
