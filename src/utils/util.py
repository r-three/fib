import argparse
import logging
import re
import random
import datetime
import os
import subprocess
import numpy as np
import torch

from shutil import copytree, ignore_patterns
from src.utils.CONSTANTS import NULL_STRING


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value


def update_dict_val_store(dict_val_store, dict_update_val):

    if dict_val_store is None:
        dict_val_store = {}
        for k in dict_update_val.keys():
            dict_val_store[k] = dict_update_val[k].detach().cpu().item()
    else:
        for k in dict_val_store.keys():
            dict_val_store[k] += dict_update_val[k].detach().cpu().item()

    return dict_val_store

def get_avg_dict_val_store(config, dict_val_store, eval_every):

    dict_avg_val = {}

    for k in dict_val_store.keys():
        old_val = dict_val_store[k]
        dict_avg_val[k] = float('%.3f' % (old_val / eval_every))
        dict_val_store[k] = 0

    return dict_val_store, dict_avg_val

def save_gcp(filepath):
    subprocess.call(f"gsutil -m -o GSUtil:parallel_composite_upload_threshold=150M \
        cp -r {filepath} \
        gs://abs_sum/{filepath}", shell=True)

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_dir(dir_name):
    '''
    Makes a directory if it doesn't exists yet
    Args:
        dir_name: directory name
    '''
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def make_exp_dir(base_exp_dir):
    '''
    Makes an experiment directory with timestamp
    Args:
        base_output_dir_name: base output directory name
    Returns:
        exp_dir_name: experiment directory name
    '''
    now = datetime.datetime.now()
    ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                            now.second)
    exp_dir_name = os.path.join(base_exp_dir, ts)
    make_dir(exp_dir_name)

    src_file = os.path.join(exp_dir_name, 'src')

    copytree(os.path.join(os.environ['LFQA_FAC_ROOT'], "src"), src_file,  ignore=ignore_patterns('*.pyc', 'tmp*'))

    return exp_dir_name

def reduce_gatheredOutput(listOfDict):
    '''
    Reduces the output from multiple devices to have the same format as for a single device.
    Also, removes the NULL datapoint, which is a dummy payload to handle model parallelism.

    Args:
        listOfDict:

    Returns:

    '''
    dictOfList = {}

    # Form list of values at each key
    for iterate_dict in listOfDict:

        # Find indices of NULL datapoint to ignore later
        idx_toRemove = {}
        for idx, datapoint_input in enumerate(iterate_dict["input"]):
            if datapoint_input == NULL_STRING:
                idx_toRemove[idx] = True

        for (k, v) in iterate_dict.items():

            # Filter out NULL datapoints based on indices.
            filtered_v = []
            for idx, datapoint_v in enumerate(v):
                if idx not in idx_toRemove:

                    filtered_v.append(datapoint_v)

            if k in dictOfList:
                dictOfList[k].append(filtered_v)
            else:
                dictOfList[k] = [filtered_v]

    # Flatten lists of list to form a list, or concatenate list of tensors to form a tensor
    for (k, batch_ofValues) in dictOfList.items():
        dictOfList[k] = [item for sublist in batch_ofValues for item in sublist]

    return dictOfList


def get_value_from_key_matching_regex(dict_regex_keyToValue, key_toMatch):
    matching_value = None
    for regex_key, value in dict_regex_keyToValue.items():
        if re.search(regex_key, key_toMatch) is not None:
            matching_value = value
    return matching_value

def get_mulChoice_outputDir(mulChoice_fp, model_name, ignore_pointMutualInfo, ignore_lengthNormalization):
    '''
    Get output dir, where we assume the filepath of the multiple choice dataset is of the
    format data/{}.jsonl where we flatten all subdirectories
    Args:
        mulChoice_fp:
        model_name:
    Returns:
    '''
    mulChoice_datasetName = mulChoice_fp\
                            .replace("multiple_choice-dataset/", "")\
                            .replace(".jsonl", "")
    model_name = model_name.replace("/fruitbasket/models/", "").replace("/", "-")
    output_dir = os.path.join("exp_out", "multiple_choice", mulChoice_datasetName)

    if ignore_pointMutualInfo:
        ignorePointMutualInfo_str = "-ignore_pointwise_mutual_info"
    else:
        ignorePointMutualInfo_str = ""

    if ignore_lengthNormalization:
        ignoreLengthNormalizationInfo_str = "-ignore_length_normalization"
    else:
        ignoreLengthNormalizationInfo_str = ""

    output_dir = os.path.join(output_dir, model_name + ignorePointMutualInfo_str + ignoreLengthNormalizationInfo_str)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir
