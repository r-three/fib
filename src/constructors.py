import re
import torch
import logging

from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from src.utils.util import get_value_from_key_matching_regex
from src.models.EncoderDecoderWrappers_forMulChoice import EncoderDecoderWrappers_forMulChoice
from src.models.DecoderWrappers_forMulChoice import DecoderWrappers_forMulChoice
from src.models.model_flags import DICT_REGEX_OF_MODEL_TYPE, DICT_REGEX_OF_DEVICE_MAP, DICT_REGEX_OF_TOKENIZERS


def log_parameter_count(model):
    total_numParam = 0
    for name, parameter in model.named_parameters():
        total_numParam += parameter.numel()
    logging.info(f"Total number of parameters in model: {total_numParam}")


def construct_hugFace_objects(model_name, max_seq_len):
    '''
    
    
    Args:
        model_name: 
        max_seq_len: 

    Returns:
        transformer:
        hugFaceConfig_forModel:
        tokenizer:
        input_prefix: Depends on how model was trained.
    '''
    tokenizer = get_value_from_key_matching_regex(DICT_REGEX_OF_TOKENIZERS, model_name)(model_name)
    tokenizer.max_seq_len = max_seq_len

    hugFaceConfig_forModel = AutoConfig.from_pretrained(model_name)

    # If model config has no input prefix, then we ignore it
    if hasattr(hugFaceConfig_forModel, "task_specific_params") and \
            hugFaceConfig_forModel.task_specific_params is not None and \
            "summarization" in hugFaceConfig_forModel.task_specific_params and \
            "prefix" in hugFaceConfig_forModel.task_specific_params["summarization"]:
        if "flan" not in model_name:
            input_prefix = hugFaceConfig_forModel.task_specific_params["summarization"]["prefix"]
            logging.info('Input Prefix: '+input_prefix)
        else:
            input_prefix = None
            logging.info('Evaluating FLAN but ignoring prompt')
    else:
        input_prefix = None

    return hugFaceConfig_forModel, tokenizer, input_prefix

def construct_models(model_name, use_hugFace_parallelism, use_bitsandbytes):

    model_type = get_value_from_key_matching_regex(DICT_REGEX_OF_MODEL_TYPE, model_name)
    device_map = get_value_from_key_matching_regex(DICT_REGEX_OF_DEVICE_MAP, model_name)
    logging.info('Model Type: ' + model_type)
    logging.info('Loading Model : ' + model_name)

    if model_type == "encoder_decoder":
        if use_hugFace_parallelism:
            logging.info('Using HuggingFace Parallelism')
            assert use_bitsandbytes == False
            transformer = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device_map)
            logging.info(transformer.hf_device_map)
        elif use_bitsandbytes:
            logging.info('Using BitsAndBytes')
            assert use_hugFace_parallelism == False
            transformer = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device_map, load_in_8bit=True)
            logging.info(transformer.hf_device_map)
        else:
            transformer = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = EncoderDecoderWrappers_forMulChoice(transformer)
    else:
        assert model_type == "decoder"
        if use_hugFace_parallelism:
            logging.info('Using HuggingFace Parallelism')
            assert use_bitsandbytes == False
            transformer = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
            logging.info(transformer.hf_device_map)
        elif use_bitsandbytes:
            logging.info('Using BitsAndBytes')
            assert use_hugFace_parallelism == False
            transformer = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, load_in_8bit=True)
            logging.info(transformer.hf_device_map)
        else:
            transformer = AutoModelForCausalLM.from_pretrained(model_name)
        model = DecoderWrappers_forMulChoice(transformer)

    log_parameter_count(transformer)

    return model, transformer