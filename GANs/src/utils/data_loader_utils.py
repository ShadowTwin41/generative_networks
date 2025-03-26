from monai.transforms.transform import MapTransform
from monai.transforms.transform import Transform
from monai.utils.enums import TransformBackends
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from collections.abc import Callable, Hashable, Mapping
import torch
import numpy as np
import time

class ConvertToMultiChannel(Transform):
    """
    Convert labels to multi channels
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        background, segmentation = (img == 0) | (img == 1), img == 1

        ### Adding noise to the background label ###
        background = background.numpy().astype(float)
        background[background==-0] = -1
        # Create a mask for elements equal to 1.0
        mask = (background == 1.0)

        # Replace the values in the image where the mask is True with noise
        random_values = np.random.normal(loc=0.0, scale=0.5, size=background.shape)
        background[mask] = random_values[mask]

        background_noise = torch.from_numpy(background).to(img.dtype)

        return torch.stack([background_noise, segmentation], dim=0) if isinstance(img, torch.Tensor) else np.stack([background_noise, segmentation], axis=0)

class ConvertToMultiChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`ConvertToMultiChannel`
    """

    backend = ConvertToMultiChannel.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannel()


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

class ReplaceLabelByNoise(Transform):
    """
    Replace the label of value 1 with noise
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        img_np = img.numpy()
        
        # Create a mask for elements equal to 0.0
        mask = (img_np == 0.0)

        # Replace the values in the image where the mask is True with noise
        random_values = np.random.normal(loc=0.0, scale=0.5, size=img_np.shape)
        img_np[mask] = random_values[mask]

        result = torch.from_numpy(img_np).to(img.dtype).unsqueeze(0)
        return result

class ReplaceLabelByNoised(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`ReplaceLabelByNoise`
    """

    backend = ReplaceLabelByNoise.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ReplaceLabelByNoise()


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

#######
####### CT head an neck cancer dataset
####### Converts the background into with and without CONTRAST
#######
class ConvertToMultiChannel_BackandForeground_Contrast(Transform):

    #Convert labels to multi channels. The backgorund with value 0 and the foreground with value 1
    

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    
    def __call__(self, img: NdarrayOrTensor, contrast) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        background, segmentation = (img == 0) | (img == 1), img == 1

        zeros_tensor = torch.zeros_like(segmentation)

        if contrast == 0:
            return torch.stack([background, zeros_tensor, segmentation], dim=0) if isinstance(img, torch.Tensor) else np.stack([background, zeros_tensor, segmentation], axis=0)
        elif contrast == 1:
            return torch.stack([zeros_tensor, background, segmentation], dim=0) if isinstance(img, torch.Tensor) else np.stack([zeros_tensor, background, segmentation], axis=0)

class ConvertToMultiChannel_BackandForeground_Contrastd(MapTransform):
    
    #Dictionary-based wrapper of :py:class:`ConvertToMultiChannel_BackandForeground_Contrast`


    backend = ConvertToMultiChannel_BackandForeground_Contrast.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannel_BackandForeground_Contrast()


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key], d["contrast"])
        return d

#########
######### For the CT head and neck cancer dataset
######### Does NOT consider CONTRAST
######### The ROI is noise, and not just the value 1
#########
class ConvertToMultiChannel_BackandForeground_noise(Transform):
    """
    Convert labels to multi channels. The backgorund with value 0 and the foreground with value 1
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        background, segmentation = (img == 0) | (img == 1), img == 1
        noise = torch.normal(mean=0.0, std=0.5, size=background.shape)

        noise_background = background * noise

        return torch.stack([noise_background, segmentation], dim=0) if isinstance(img, torch.Tensor) else np.stack([noise_background, segmentation], axis=0)

class ConvertToMultiChannel_BackandForeground_noised(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`ConvertToMultiChannel_BackandForeground_noise`
    """

    backend = ConvertToMultiChannel_BackandForeground_noise.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannel_BackandForeground_noise()


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

#########
######### For the CT head and neck cancer dataset
######### Does NOT consider CONTRAST
######### The ROI is blank (value 1)
#########
class ConvertToMultiChannel_BackandForeground_blank(Transform):
    """
    Convert labels to multi channels. The backgorund with value 0 and the foreground with value 1
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        background, segmentation = (img == 0) | (img == 1), img == 1

        return torch.stack([background, segmentation], dim=0) if isinstance(img, torch.Tensor) else np.stack([background, segmentation], axis=0)

class ConvertToMultiChannel_BackandForeground_blankd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`ConvertToMultiChannel_BackandForeground_blank`
    """

    backend = ConvertToMultiChannel_BackandForeground_blank.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannel_BackandForeground_blank()


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

class ConvertToMultiChannelBasedOnBratsClasses2023(Transform):
    """
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [(img == 1) | (img == 3), (img == 1) | (img == 2) | (img == 3), img == 3]
        # merge labels 1 (tumor non-enh) and 3 (tumor enh) and 2 (large edema) to WT
        # label 3 is ET
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)


class ConvertToMultiChannelBasedOnBratsClasses2023d(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses2023`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 3 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = ConvertToMultiChannelBasedOnBratsClasses2023.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnBratsClasses2023()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class ScaleIntensityRange_Tanh(Transform):
    """
    Apply specific intensity scaling to the whole numpy array.

    Args:
        a_min: intensity original range min.
        a_max: intensity original range max.
        clip: whether to perform clip after scaling.
        dtype: output data type, if None, same as input image. defaults to float32.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, a_min: float, a_max: float) -> None:
        self.a_min = a_min
        self.a_max = a_max

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        #img = convert_to_tensor(img, track_meta=get_track_meta())
        img = np.clip(img, self.a_min, self.a_max)
        img = np.tanh(0.02 * img)

        return img


class ScaleIntensityRanged_Tanh(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensityRange_Tanh`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        a_min: intensity original range min.
        a_max: intensity original range max.
        b_min and b_max: compatibility purposes.
        clip: whether to perform clip before scaling.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = ScaleIntensityRange_Tanh.backend

    def __init__(self, keys:KeysCollection, a_min:float, a_max:float, b_min:float, b_max:float, clip:bool, allow_missing_keys:bool=False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ScaleIntensityRange_Tanh(a_min, a_max)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d