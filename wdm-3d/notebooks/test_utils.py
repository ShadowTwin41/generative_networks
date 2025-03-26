import multiprocessing
import os
from copy import deepcopy
from multiprocessing import Pool
import numpy as np
import torch
from monai.transforms import (
    Compose, 
    LoadImaged,
    EnsureChannelFirstd, 
    EnsureTyped,
    Orientationd,
    ScaleIntensityRanged, 
    ResizeWithPadOrCropd,
    CopyItemsd,
    )

from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, isfile
import sys
sys.path.insert(1, "..")
from utils.data_loader_utils import ConvertToMultiChannelBasedOnBratsClasses2023d, QuantileAndScaleIntensityd
from monai.data import DataLoader, CacheDataset
from nnunetv2.utilities.json_export import recursive_fix_for_json_export

def label_or_region_to_key(label_or_region):
    return str(label_or_region)

def save_summary_json(results: dict, output_file: str):
    """
    json does not support tuples as keys (why does it have to be so shitty) so we need to convert that shit
    ourselves
    """
    results_converted = deepcopy(results)
    # convert keys in mean metrics
    results_converted['mean'] = {label_or_region_to_key(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results_converted["metric_per_case"])):
        results_converted["metric_per_case"][i]['metrics'] = \
            {label_or_region_to_key(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    # sort_keys=True will make foreground_mean the first entry and thus easy to spot
    save_json(results_converted, output_file, sort_keys=True)
    
def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn

def region_or_label_to_mask(segmentation, region_or_label) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask

def compute_metrics(reference_file, prediction_file, image_reader_writer, labels_or_regions, ignore_label) -> dict:
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}
    for r in labels_or_regions:
        results['metrics'][r] = {}
        mask_ref = region_or_label_to_mask(seg_ref, r)
        mask_pred = region_or_label_to_mask(seg_pred, r)
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
        if np.sum(mask_ref)==0 and  np.sum(mask_pred)==0:
            results['metrics'][r]['Dice'] = 1
            results['metrics'][r]['IoU'] = 1
        elif tp + fp + fn == 0:
            results['metrics'][r]['Dice'] = np.nan
            results['metrics'][r]['IoU'] = np.nan
        else:
            results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
            results['metrics'][r]['IoU'] = tp / (tp + fp + fn)
        results['metrics'][r]['FP'] = fp
        results['metrics'][r]['TP'] = tp
        results['metrics'][r]['FN'] = fn
        results['metrics'][r]['TN'] = tn
        results['metrics'][r]['n_pred'] = fp + tp
        results['metrics'][r]['n_ref'] = fn + tp
    return results

def compute_metrics_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                              image_reader_writer,
                              file_ending,
                              regions_or_labels,
                              ignore_label,
                              num_processes,
                              chill) -> dict:

    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_ref exist in folder_pred"
    files_ref = [join(folder_ref, i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        results = pool.starmap(
            compute_metrics,
            list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred),
                     [ignore_label] * len(files_pred)))
        )

    # mean metric per class
    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())
    means = {}
    for r in regions_or_labels:
        means[r] = {}
        for m in metric_list:
            means[r][m] = {"mean": np.nanmean([i['metrics'][r][m] for i in results]),
                           "std": np.nanstd([i['metrics'][r][m] for i in results])}

    # foreground mean
    foreground_mean = {}
    for m in metric_list:
        values = []
        for k in means.keys():
            if k == 0 or k == '0':
                continue
            values.append(means[k][m]["mean"])
        foreground_mean[m] = {"mean": np.mean(values), "std": np.std(values)}

    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    recursive_fix_for_json_export(foreground_mean)
    result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}
    if output_file is not None:
        save_summary_json(result, output_file)
    return result
   
def absolute_mean_error(y_true, y_pred):
    """
    Compute the absolute mean error (MAE) between two arrays.

    Parameters:
    y_true (numpy.ndarray): The true values.
    y_pred (numpy.ndarray): The predicted or estimated values.

    Returns:
    float: The absolute mean error (MAE).
    """
    # Calculate the absolute differences between the true and predicted values
    abs_errors = np.abs(y_true - y_pred)

    # Compute the mean of the absolute differences
    mean_absolute_error = np.mean(abs_errors)

    return round(mean_absolute_error,2)

def get_data_loader(image_key, clip_min, clip_max, data_list, two_loaders, pad_or_crop, cache_rate=[0,1], not_normalise=False):

    b_min = float(-1.0)
    b_max = float(1.0)
    
    train_transforms = [
        LoadImaged(keys=[image_key], meta_key_postfix="meta_dict", image_only=False),
        EnsureChannelFirstd(keys=[image_key]),
        EnsureTyped(keys=[image_key], dtype=torch.float32),
        Orientationd(keys=[image_key], axcodes="RAS")        
    ]
    if not not_normalise:
        train_transforms.append(ScaleIntensityRanged(keys=[image_key], a_min=float(clip_min), a_max=float(clip_max), b_min=float(b_min), b_max=float(b_max), clip=True))
    else:
        print("Not normalising")
    if pad_or_crop:
        train_transforms.append(ResizeWithPadOrCropd(
                keys=[image_key],
                spatial_size=(256,256,256),
                mode="constant",
                value=clip_min # The value was -1 originally
            ))
    train_transforms.append(EnsureTyped(keys=[image_key], dtype=torch.float32))
    train_transforms_final =  Compose(train_transforms)

    # Creating traing dataset
    ds = CacheDataset( 
        data=data_list, 
        transform=train_transforms_final,
        cache_rate=cache_rate[0], 
        copy_cache=False,
        progress=True,
        num_workers=4,
    )

    dataloader_1 = DataLoader(ds, batch_size=1, shuffle=False)

    if two_loaders:

        ds = CacheDataset( 
            data=data_list, 
            transform=train_transforms_final,
            cache_rate=cache_rate[1], 
            copy_cache=False,
            progress=True,
            num_workers=4,
        )

        dataloader_2 = DataLoader(ds, batch_size=1, shuffle=False)
        return dataloader_1, dataloader_2
    else:
        return dataloader_1



from monai.transforms.transform import MapTransform
from monai.transforms.transform import Transform
from monai.utils.enums import TransformBackends
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from collections.abc import Callable, Hashable, Mapping
from monai.config import DtypeLike
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor, get_equivalent_dtype

class OnlyScaleIntensity(Transform):
    """
    Apply range scaling to a numpy array based on the intensity distribution of the input.

    Args:
        a_min: intensity target range min.
        a_max: intensity target range max.
        dtype: output data type, if None, same as input image. defaults to float32.
    """

    def __init__(self) -> None:
        pass

    def _normalize(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        normalize=(lambda x: 2*x - 1)
        out_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
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

class OnlyScaleIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.QuantileAndScaleIntensity`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        a_min: intensity target range min.
        a_max: intensity target range max.
        relative: whether to scale to the corresponding percentiles of [a_min, a_max]
        channel_wise: if True, compute intensity percentile and normalize every channel separately.
            default to False.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = OnlyScaleIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys=False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = OnlyScaleIntensity()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d


def get_BraTS_data_loader(in_keys, data_list, cache_rate, apply_quantile):
    train_transforms = [
                    LoadImaged(keys=in_keys, meta_key_postfix="meta_dict", image_only=False),
                    EnsureChannelFirstd(keys=in_keys),
                    EnsureTyped(keys=in_keys, dtype=torch.float32),
                    Orientationd(keys=in_keys, axcodes="RAS")
                ]
    if apply_quantile:
        train_transforms.append(QuantileAndScaleIntensityd(keys=in_keys))
    else:
        train_transforms.append(OnlyScaleIntensityd(keys=in_keys))

    train_transforms.append(EnsureTyped(keys=in_keys, dtype=torch.float32))
    train_transforms = Compose(train_transforms)
    # Creating traing dataset
    ds = CacheDataset( 
        data=data_list,
        transform=train_transforms,
        cache_rate=cache_rate, 
        copy_cache=False,
        progress=True,
        num_workers=4,
    )
    
    # Creating data loader
    dl = DataLoader(
        ds,
        batch_size=1,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        shuffle=False, 
        #collate_fn=no_collation,
    )
    return dl
