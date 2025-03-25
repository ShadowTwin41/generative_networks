import numpy as np
import torch
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Transform
from monai.utils.enums import TransformBackends
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import MapTransform
from collections.abc import Callable, Hashable, Mapping


class ConvertHeadNNeckCancer(Transform):
    """
    Convert labels to multi channels based on head and neck classes:
    label 1 -> GTV
    Return:
       GTV region
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [img == 1]
        # label 1 -> GTV 
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)

class ConvertHeadNNeckCancerd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertHeadNNeckCancer`.
    Convert labels to multi channels based on head and neck classes:
    label 1 -> GTV
    Return:
        GTV region
    """

    backend = ConvertHeadNNeckCancer.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertHeadNNeckCancer()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

