import torch
import json
from torch.utils import data
import copy
import math
import re
import logging

from src.utils.CONSTANTS import NULL_STRING
from src.data.templates import SUMMARIZATION_PROMPT_TEMPLATES
from src.data.preprocess_data import tokenize_prompted_input_text, does_tokenizer_addBosEosTokens

class MultipleChoiceReader(object):
    '''
    MultipleChoiceReader reads any multiple choice dataset
    '''

    def read_mulChoiceData(self, mulChoiceFilepath):
        '''
        Read dataset

        Args:
            mcFilmulChoiceFilepathepath:

        Returns:
            listOf_MCDatapoints:
        '''
        fd = open(mulChoiceFilepath, 'r')

        listOfDatapoints = []
        for line in fd.readlines():
            datapoint = json.loads(line)
            listOfDatapoints.append(datapoint)

        return listOfDatapoints

NULL_DATAPOINT = {
    "id": NULL_STRING,
    "input": NULL_STRING,
    "list_choices": [NULL_STRING, NULL_STRING],
    "correct_choice": NULL_STRING,
    "lbl": 0
}


class MultipleChoiceDataset(data.Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 promptTemplate_idx,
                 input_prefix,
                 device,
                 world_size):

        # if the device is an integer, then this means we are using parallelism and have to split the dataset among each device.
        if isinstance(device, int):

            num_datapoints_per_split = math.ceil(len(data) / world_size)

            device_data = []
            for idx, datapoint in enumerate(data):
                if idx % world_size == device:
                    device_data.append(datapoint)

            # We ensure each device sees the same number of samples, so that the number of batches is same per device.
            # If the batch size is 1, and world_size=2, then number of batches will be different per device.
            # This will cause a race condition for parallelism.
            if len(device_data) < num_datapoints_per_split:
                device_data.append(NULL_DATAPOINT)
                assert len(device_data) == num_datapoints_per_split
            self.data = device_data
        # For non-parallelism
        else:
            self.data = data

        self.tokenizer = tokenizer

        # Uses no template and adds the input prefix only.
        # Note that the 0 template is just the data.
        if input_prefix is not None:
            assert promptTemplate_idx == 0
            self.prompt_template = input_prefix + SUMMARIZATION_PROMPT_TEMPLATES[0]

        # Create template from prompt template idx
        else:
            self.prompt_template = SUMMARIZATION_PROMPT_TEMPLATES[promptTemplate_idx]

            if promptTemplate_idx == 0:
                # If the tokenizer does not insert a BOS or EOS token for an empty string, we need to add an empty space
                #   so that we can have a null input when computing PMI. This holds for BLOOM.
                # This only has to be done for the zero prompt since there is no additional text in the prompt.
                # Though bloom was not pretrained to insert this empty space, it should not affect performance much.
                if len(tokenizer("")["input_ids"]) == 0:
                    self.prompt_template = " " + self.prompt_template

        logging.info('Prompt Template: '+self.prompt_template)
        self.device = device
        self.add_bosToken, self.add_eosToken = does_tokenizer_addBosEosTokens(self.tokenizer)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, get_idx):
        datapoint = self.data[get_idx]

        input_ids, input_masks, input_txt, nullInput_txt = tokenize_prompted_input_text(self.tokenizer,
                                                                                       datapoint,
                                                                                       self.prompt_template,
                                                                                       self.add_bosToken,
                                                                                       self.add_eosToken)
        nullInput_dict = self.tokenizer(nullInput_txt,
                                           return_tensors="pt",
                                           truncation=True)
        nullInput_ids = nullInput_dict["input_ids"][0]
        nullInput_masks = nullInput_dict["attention_mask"][0]

        allChoices_ids = []
        allChoices_masks = []

        for choice in datapoint["list_choices"]:
            choiceDict = self.tokenizer(choice, return_tensors="pt", truncation=True)
            # Skip BOS token for choices since it is a continuation of the input
            # TODO Currently this assumes that a BOS token is not added for encoder-decoder models.
            # TODO add logic to NOT ignore the BOS token in the choices for encoder-decoder model
            # Note that all T5 variants do not add a BOS token.
            if self.add_bosToken:
                start_idx = 1
            else:
                start_idx = 0
            allChoices_ids.append(choiceDict["input_ids"][0][start_idx:])
            allChoices_masks.append(choiceDict["attention_mask"][0][start_idx:])

        return {"id": datapoint["id"],
                "input": input_txt,
                "input_ids": input_ids,
                "input_masks": input_masks,
                "null_input_ids": nullInput_ids,
                "null_input_masks": nullInput_masks,
                "list_choices": datapoint["list_choices"],
                "all_choices_ids": allChoices_ids,
                "all_choices_lbls": copy.deepcopy(allChoices_ids),
                "all_choices_masks": allChoices_masks,
                "correct_choice": datapoint["correct_choice"],
                "lbl": datapoint["lbl"]}

    def collate_fn(self, batch_ofDatapoints):
        '''
        Convert a batch of datapoints into a datapoint that is batched.  This is meant to
        override the default collate function in pytorch.

        Args:
            batch_ofDatapoints:

        Returns:

        '''
        datapoint_batched = {}

        for datapoint in batch_ofDatapoints:
            for (k, v) in datapoint.items():
                if k in datapoint_batched:
                    # Each value in all_choices is already a list, so we extend and not append.
                    if "all_choices" in k:
                        datapoint_batched[k].extend(v)
                    else:
                        datapoint_batched[k].append(v)
                else:
                    # Each value in all_choices is already a list, so we do not need to
                    # initialize a list with v in it, and can just use v.
                    if "all_choices" in k:
                        datapoint_batched[k] = v
                    else:
                        datapoint_batched[k] = [v]

        for (k, batch_ofValues) in datapoint_batched.items():
            # If id or mask is in key, this means we need to pad to the longest sequence length
            if ("ids" in k) or ("masks" in k) or (k == "all_choices_lbls"):
                if "ids" in k:
                    padToken_id = self.tokenizer.pad_token_id
                    if padToken_id is None:
                        padToken_id = self.tokenizer.eos_token_id
                elif "masks" in k:
                    padToken_id = 0
                elif k == "all_choices_lbls":
                    padToken_id = -100
                else:
                    raise ValueError(f"The key {k} has ids or masks but is not recognized")
                datapoint_batched[k] = torch.nn.utils.rnn.pad_sequence(
                    batch_ofValues,
                    batch_first=True,
                    padding_value=padToken_id)

                if self.device is not None:
                    datapoint_batched[k] = datapoint_batched[k].to(self.device)

            elif isinstance(batch_ofValues[0], int):
                datapoint_batched[k] = torch.tensor(batch_ofValues)

                if self.device is not None:
                    datapoint_batched[k] = datapoint_batched[k].to(self.device)



        return datapoint_batched


