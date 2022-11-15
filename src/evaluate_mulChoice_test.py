import os
import argparse
import torch
import logging
import re
from tqdm import tqdm
import copy
import json


from transformers import AutoTokenizer, GPT2Tokenizer

from src.multiple_choice.utils import read_jsonl, write_jsonl
from src.data.preprocess_data import does_tokenizer_addBosEosTokens

from src.models.model_flags import DICT_REGEX_OF_MODEL_TYPE, DICT_REGEX_OF_DEVICE_MAP
from src.utils.util import get_value_from_key_matching_regex


def compute_totalLogProb(logProb_ofIds):
    '''
       Assumes log probs will be zero for ids which are supposed to be masked out, ie. pad tokens

       Args:
           logProb_ofIds:

       Returns:

       '''
    return torch.sum(logProb_ofIds, dim=1)

def compute_avgLogProb(logProb_ofIds, max_len):
    '''
    Assumes log probs will be zero for ids which are supposed to be masked out, ie. pad tokens

    Args:
        logProb_ofIds:

    Returns:

    '''
    return torch.sum(logProb_ofIds, dim=1) / max_len

def get_model(prediction_filepath):
    listOf_directories = prediction_filepath.split("/")
    return listOf_directories[-2]

def check_scoreMatches_recomputingFromCache(predictionPrompt_filepath):
    list_json = read_jsonl(predictionPrompt_filepath)

    for json in list_json:
        logProb_ofAllChoiceIds = torch.tensor(json["log_probs_of_all_choices_ids"])
        score_ofChoices = torch.tensor(json["score_of_choices"])
        logProb_ofAllChoiceIds_condNullInput = torch.tensor(json["log_prob_of_all_choices_ids_cond_null_input"])
        len_allChoices = torch.tensor(json["len_all_choices"])
        pred_choice = json["pred_choice"]

        allChoices_logProb = compute_avgLogProb(logProb_ofAllChoiceIds, len_allChoices)
        allChoices_logProb_condNullInput = compute_avgLogProb(logProb_ofAllChoiceIds_condNullInput, len_allChoices)
        allChoices_logProb -= allChoices_logProb_condNullInput

        if not torch.allclose(allChoices_logProb, score_ofChoices, atol=1e-4):
            print(predictionPrompt_filepath)
            import ipdb; ipdb.set_trace()

        # Handle case where predicted probabilities are same for both choices, so
        # argmax might not be consistent
        if not torch.argmax(allChoices_logProb) == pred_choice and\
            not torch.allclose(allChoices_logProb[0], allChoices_logProb[1], atol=1e-4):
            print(predictionPrompt_filepath)
            import ipdb; ipdb.set_trace()

dictOfModel_toDictOfInput_toGoldSummaryLogProb = {}
def check_correctSummaryScoreMatches_acrossDifferentDistractors(predictionPrompt_filepath):
    list_json = read_jsonl(predictionPrompt_filepath)

    model = get_model(predictionPrompt_filepath)

    if model not in dictOfModel_toDictOfInput_toGoldSummaryLogProb:
        dictOfModel_toDictOfInput_toGoldSummaryLogProb[model] = {}

    for json in list_json:
        score_ofChoices = torch.tensor(json["score_of_choices"])
        correctChoice_logProb = score_ofChoices[json["lbl"]]
        input = json["input"]

        if input in dictOfModel_toDictOfInput_toGoldSummaryLogProb[model]:
            if not torch.allclose(dictOfModel_toDictOfInput_toGoldSummaryLogProb[model][input][0], correctChoice_logProb, atol=1e-4):
                print(predictionPrompt_filepath)
        else:
            dictOfModel_toDictOfInput_toGoldSummaryLogProb[model][input] = correctChoice_logProb, json["list_choices"][json["lbl"]]

def check_accuraciesCorrect_andExistsForEachPrompt(mulChoice_experiment):
    '''


    Returns:

    '''
    scores_filepath = os.path.join(mulChoice_experiment, "scores.jsonl")

    # Check there are 3 scores for 3 prompts
    if os.path.exists(scores_filepath):
        list_scoresJson = read_jsonl(scores_filepath)
        if len(list_scoresJson) != 3:
            # t5-large finetuned only use 1 prompt, so it should have 1 score
            assert len(list_scoresJson) == 1 and \
                   ("sysresearch101-t5-large-finetuned-xsum" in mulChoice_experiment),\
            mulChoice_experiment
    else:
        print(scores_filepath)
        import ipdb; ipdb.set_trace()

    for score_json in list_scoresJson:
        prompt_idx = score_json["prompt_template_idx"]
        predictionPrompt_filepath = os.path.join(mulChoice_experiment, f"predictions-prompt_{prompt_idx}.json")

        list_json = read_jsonl(predictionPrompt_filepath)
        num_correct = 0
        for json in list_json:
            if json["pred_choice"] == json["lbl"]:
                num_correct += 1

        computed_acc = round(num_correct / len(list_json), 3)
        assert computed_acc == score_json["multiple-choice-accuracy"]

def test_experiment(mulChoice_experiment):
    check_accuraciesCorrect_andExistsForEachPrompt(mulChoice_experiment)

    for prompt_idx in range(3):
        predictionPrompt_filepath = os.path.join(mulChoice_experiment, f"predictions-prompt_{prompt_idx}.json")

        if os.path.exists(predictionPrompt_filepath):
            check_scoreMatches_recomputingFromCache(predictionPrompt_filepath)
            check_correctSummaryScoreMatches_acrossDifferentDistractors(predictionPrompt_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--list_mulChoiceExperiments", action='store', type=str, nargs='*', required=True)
    args = parser.parse_args()

    for mulChoice_experiment in tqdm(args.list_mulChoiceExperiments):
        test_experiment(mulChoice_experiment)
