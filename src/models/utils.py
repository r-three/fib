import torch

def compute_logProb(logProbs_ofAllChoices_ids, allChoices_masks, num_choices, maxChoice_len, lengthNormalization):
    '''


    Args:
        logProbs_forAllChoices_ids: [batch_size x num_choices x max_choice_len]
        allChoices_masks: [batch_size, num_choices, max_choice_len]
        num_choices:
        maxChoice_len:
        lengthNormalization:

    Returns:
        logProbs_ofAllChoices: [batch_size, num_choices]
        logProbs_ofAllChoicesIds_zeroOutPadIds: [batch_size, num_choices, max_choice_len]
        len_allChoices: [batch_size ]
    '''
    # Compute the log probabilities of all the choices by averaging the log probabilities of
    # the ids and zeroing out the pad ids
    # [batch_size, num_choices, max_choice_len]
    logProbs_ofAllChoices_ids = logProbs_ofAllChoices_ids.reshape(-1, num_choices, maxChoice_len)
    allChoices_masks = allChoices_masks.reshape(-1, num_choices, maxChoice_len) > 0
    logProbs_ofAllChoicesIds_zeroOutPadIds = logProbs_ofAllChoices_ids * allChoices_masks
    logProbs_ofAllChoices = torch.sum(logProbs_ofAllChoicesIds_zeroOutPadIds, dim=2)
    len_allChoices = torch.sum(allChoices_masks, dim=2)

    if lengthNormalization:
        logProbs_ofAllChoices = logProbs_ofAllChoices / len_allChoices

    return logProbs_ofAllChoices,\
           logProbs_ofAllChoicesIds_zeroOutPadIds, \
           len_allChoices
