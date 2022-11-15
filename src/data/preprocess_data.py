import re
import torch


def does_tokenizer_addBosEosTokens(tokenizer):
    # Compute whether to add BOS or EOS tokens by tokenizing a dummy input.
    filler_ids = tokenizer("hello")["input_ids"]
    add_bosToken = False
    if filler_ids[0] == tokenizer.bos_token_id:
        add_bosToken = True

    add_eosToken = False
    if filler_ids[-1] == tokenizer.eos_token_id:
        add_eosToken = True

    return add_bosToken, add_eosToken

def tokenize_prompted_input_text(tokenizer, datapoint, prompt_template, add_bosToken, add_eosToken):
    '''
    Gets the input text and tokenizes it from prompt.

    Assumes the datapoint is a dictionary and the prompt template specifies
    which value of the datapoint to use based on the key wrapped in [].
    For example, [input] should be used to specify the input.
    Note, the prompt template cannot use [] in any other locations.


    Args:
        tokenizer:
        datapoint:
        prompt_template:
        add_bosToken:
        add_eosToken:

    Returns:

    '''
    template_nonDataKeys = re.split(r"\[.*\]", prompt_template)
    template_dataKeys = re.findall(r"\[.*\]", prompt_template)

    assert len(template_nonDataKeys) == len(template_dataKeys) + 1

    remaining_seqLen = tokenizer.max_seq_len
    num_dataKeys = len(template_dataKeys)

    list_nonDataKeys_txt = []
    list_nonDataKeys_ids = []
    list_nonDataKeys_mask = []

    for nonDataKey in template_nonDataKeys:
        if len(nonDataKey) > 0:
            list_nonDataKeys_txt.append(nonDataKey)
            nonDataKey_dict = tokenizer(nonDataKey, add_special_tokens=False)
            list_nonDataKeys_ids.append(nonDataKey_dict["input_ids"])
            list_nonDataKeys_mask.append(nonDataKey_dict["attention_mask"])
            remaining_seqLen -= len(nonDataKey_dict["input_ids"])
        else:
            list_nonDataKeys_txt.append("")
            list_nonDataKeys_ids.append([])
            list_nonDataKeys_mask.append([])

    # This list will recombine the nonDataKeys and dataKeys in the correct order.
    list_split_txt = []
    list_split_ids = []
    list_split_masks = []

    if add_bosToken:
        list_split_ids.append(tokenizer.bos_token_id)
        list_split_masks.append(1)
        remaining_seqLen -= 1

    # We have to compute remaining sequence length at the beginning
    # to know how much is left over.
    if add_eosToken:
        remaining_seqLen -= 1

    # Add any text in template that appears before the first data key.
    list_split_txt.append(list_nonDataKeys_txt[0])
    list_split_ids.extend(list_nonDataKeys_ids[0])
    list_split_masks.extend(list_nonDataKeys_mask[0])


    for i in range(num_dataKeys):
        dataKey = template_dataKeys[i].replace("[", "").replace("]", "")
        dataValue = datapoint[dataKey]

        dataValue_dict = tokenizer(dataValue, add_special_tokens=False)

        value_ids = dataValue_dict["input_ids"]
        value_mask = dataValue_dict["attention_mask"]

        len_value = len(dataValue_dict["input_ids"])
        if len_value > remaining_seqLen:
            value_txt = tokenizer.decode(value_ids[:remaining_seqLen], add_special_tokens=False)
            value_ids = value_ids[:remaining_seqLen]
            value_mask = value_mask[:remaining_seqLen]
            remaining_seqLen = 0
        else:
            value_txt = tokenizer.decode(value_ids, add_special_tokens=False)
            remaining_seqLen -= len_value

        # Add tokenized values from data
        list_split_txt.append(value_txt)
        list_split_ids.extend(value_ids)
        list_split_masks.extend(value_mask)

        # Add tokenized text between data
        # Increment by 1 since we add non-data key text at the very beginning
        list_split_txt.append(list_nonDataKeys_txt[i+1])
        list_split_ids.extend(list_nonDataKeys_ids[i+1])
        list_split_masks.extend(list_nonDataKeys_mask[i+1])

    if add_eosToken:
        list_split_ids.append(tokenizer.eos_token_id)
        list_split_masks.append(1)

    return torch.tensor(list_split_ids), torch.tensor(list_split_masks), "".join(list_split_txt), "".join(template_nonDataKeys)
