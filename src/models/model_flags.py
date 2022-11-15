from src.models.device_maps import BLOOM_DEVICE_MAP, GPT_NEOX_DEVICE_MAP, T0_DEVICE_MAP, OPT_66B_DEVICE_MAP, OPT_175B_DEVICE_MAP

from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.models.gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast

DICT_REGEX_OF_MODEL_TYPE = {
    ".*T0.*": "encoder_decoder",
    ".*pegasus.*": "encoder_decoder",
    ".*t5.*": "encoder_decoder",
    ".*bart.*": "encoder_decoder",
    ".*bloom.*": "decoder",
    ".*gpt.*": "decoder",
    ".*opt.*": "decoder",
    ".*T5.*": "encoder_decoder",
}

DICT_REGEX_OF_WHETHER_MODEL_USES_POSITION_IDS = {
    ".*bloom.*": False,
    ".*gpt.*": True,
    ".*opt.*": False
}

DICT_REGEX_OF_DEVICE_MAP = {
    ".*": "auto",
    ".*bloom": BLOOM_DEVICE_MAP,
    ".*gpt-neox-20b": GPT_NEOX_DEVICE_MAP,
    ".*T0|.*t5-xxl.*": T0_DEVICE_MAP,
    ".*opt-66b": OPT_66B_DEVICE_MAP,
    ".*opt-175b": OPT_175B_DEVICE_MAP
}

DICT_REGEX_OF_TOKENIZERS = {
    ".*": lambda model_name: AutoTokenizer.from_pretrained(model_name),
    ".*opt.*": lambda model_name: AutoTokenizer.from_pretrained(model_name, use_fast=False),
    ".*gpt-neox-20b": lambda model_name: GPTNeoXTokenizerFast.from_pretrained(model_name)
}