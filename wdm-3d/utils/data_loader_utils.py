from monai.transforms.transform import MapTransform
from monai.transforms.transform import Transform
from monai.utils.enums import TransformBackends
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from collections.abc import Callable, Hashable, Mapping
from monai.config import DtypeLike
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor, get_equivalent_dtype


import torch
import numpy as np
import time
"""
class ConvertToMultiChannel_BackandForeground_Contrast(Transform):
    
    #Replace the label of value 1 with noise
    

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

class ConvertToMultiChannel_BackandForeground_Contrastd(MapTransform):

    # Dictionary-based wrapper of :py:class:`ConvertToMultiChannel_BackandForeground_Contrast`
    

    backend = ConvertToMultiChannel_BackandForeground_Contrast.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannel_BackandForeground_Contrast()


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d
"""
#######
####### CT head an neck cancer dataset
####### Converts the background into with and without CONTRAST
#######
class ConvertToMultiChannel_BackandForeground_Contrast(Transform):

    #Convert labels to multi channels. The backgorund with value 0 and the foreground with value 1
    

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    
    def __call__(self, img: NdarrayOrTensor, contrast, no_seg, full_background) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        background, segmentation = (img == 0) | (img == 1), img == 1
        
        if full_background:
            background = torch.ones_like(background)
        
        zeros_tensor = torch.zeros_like(segmentation)
        if contrast == 0:
            if no_seg:
                return torch.stack([background, zeros_tensor], dim=0) if isinstance(img, torch.Tensor) else np.stack([background, zeros_tensor], axis=0)
            else:
                return torch.stack([background, zeros_tensor, segmentation], dim=0) if isinstance(img, torch.Tensor) else np.stack([background, zeros_tensor, segmentation], axis=0)
            
        elif contrast == 1:
            if no_seg:
                return torch.stack([zeros_tensor, background], dim=0) if isinstance(img, torch.Tensor) else np.stack([zeros_tensor, background], axis=0)
            else:
                return torch.stack([zeros_tensor, background, segmentation], dim=0) if isinstance(img, torch.Tensor) else np.stack([zeros_tensor, background, segmentation], axis=0)
            
class ConvertToMultiChannel_BackandForeground_Contrastd(MapTransform):
    
    #Dictionary-based wrapper of :py:class:`ConvertToMultiChannel_BackandForeground_Contrast`


    backend = ConvertToMultiChannel_BackandForeground_Contrast.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, no_seg=False, full_background=False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannel_BackandForeground_Contrast()
        self.no_seg = no_seg
        self.full_background = full_background

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key], d["contrast"], no_seg=self.no_seg, full_background=self.full_background)
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
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor, no_seg) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        
        result = [(img == 1) | (img == 3), (img == 1) | (img == 2) | (img == 3), img == 3]
        # TC = 1 for NCR and 3 for ET
        # WT = labels 1 (tumor non-enh) and 3 (tumor enh) and 2 (large edema) 
        # label 3 is ET
        three_channel_label = torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)
        
        if no_seg:
            three_channel_label = torch.zeros_like(three_channel_label)

        return three_channel_label


class ConvertToMultiChannelBasedOnBratsClasses2023d(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses2023`
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 3 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = ConvertToMultiChannelBasedOnBratsClasses2023.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, no_seg: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnBratsClasses2023()
        self.no_seg = no_seg
    
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key], no_seg=self.no_seg)
        return d
####
## Transform to scale intensity, following the original WDM 3D github implementation
####

class QuantileAndScaleIntensity(Transform):
    """
    Apply range scaling to a numpy array based on the intensity distribution of the input.

    Args:
        lower: lower quantile.
        upper: upper quantile.
        a_min: intensity target range min.
        a_max: intensity target range max.
        dtype: output data type, if None, same as input image. defaults to float32.
    """

    def __init__(self) -> None:
        pass

    def _normalize(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        normalize=(lambda x: 2*x - 1)
        out_clipped = np.clip(img, np.quantile(img, 0.001), np.quantile(img, 0.999))
        out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
        out_normalized= normalize(out_normalized)
        #img = convert_to_tensor(out_normalized, track_meta=False)
        return out_normalized

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        out_normalized = self._normalize(img=img)
        out = convert_to_dst_type(out_normalized, dst=img)[0]
        return out

class QuantileAndScaleIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.QuantileAndScaleIntensity`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        lower: lower quantile.
        upper: upper quantile.
        a_min: intensity target range min.
        a_max: intensity target range max.
        relative: whether to scale to the corresponding percentiles of [a_min, a_max]
        channel_wise: if True, compute intensity percentile and normalize every channel separately.
            default to False.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = QuantileAndScaleIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys=False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = QuantileAndScaleIntensity()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
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