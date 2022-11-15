import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.util import get_value_from_key_matching_regex
from src.models.model_flags import DICT_REGEX_OF_WHETHER_MODEL_USES_POSITION_IDS
from src.models.utils import compute_logProb

class DecoderWrappers_forMulChoice(nn.Module):
    '''

    '''

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

        self.use_position_ids = get_value_from_key_matching_regex(DICT_REGEX_OF_WHETHER_MODEL_USES_POSITION_IDS, self.transformer._get_name().lower())

        if "gptneox" in self.transformer._get_name().lower():
            self.use_position_ids = False
            print("WARNING! NeoX has a bug with padding the input when caching key,values. Use a batch size of 1.")
        assert self.use_position_ids is not None


    def _broadcast_tensors(self, input_masks, past_key_values, num_choices):
        '''
        Broadcast the input masks and encoder outputs to account for multiple choices per input

        Args:
            input_masks: [batch_size, max_input_len]
            past_key_values: Tuple of keys and values for each layer.
                The first index of the tuple is the layer index, and the second index
                of the tuple is whether it is a key or value. Each element in tuple
                has shape [batch_size, max_input_len, num_heads, head_dim] or [batch_size x num_heads, head_dim, max_input_len].
            num_choices:

        Returns:
            input_masks: [batch_size x num_choices, max_input_len]
            past_key_values: Tuple of keys and values for each layer.
                The first index of the tuple is the layer index, and the second index
                of the tuple is whether it is a key or value. Each element in tuple
                has shape [batch_size x num_choices, max_input_len, num_heads, head_dim]
                or [batch_size x num_heads x num_choices, head_dim, max_input_len].
        '''
        batch_size, max_input_len = input_masks.shape
        input_masks = torch.repeat_interleave(input_masks, num_choices, dim=0)

        list_broadcast_pastKeyValues = []
        for pastKeyValues_perLayer in past_key_values:

            list_broadcast_pastKeyValues_perLayer = []
            for key_or_value in pastKeyValues_perLayer:
                # This is for keys or values which have dimension [batch_size, max_input_len, num_heads, head_dim]
                # This is the standard for Hugging Face.
                if len(key_or_value.shape) == 4:
                    list_broadcast_pastKeyValues_perLayer.append(torch.repeat_interleave(key_or_value, num_choices, dim=0))
                # This is for keys or values which have dimension [batch_size x num_heads, head_dim, max_input_len].
                # This is what is used for BLOOM in transformers == 4.22.0
                elif len(key_or_value.shape) == 3:
                    num_heads = key_or_value.shape[0] // batch_size
                    flatten_keyOrValue = key_or_value.reshape(((batch_size, num_heads) + key_or_value.shape[1:]))
                    broadcast_flatten_keyOrValue = torch.repeat_interleave(flatten_keyOrValue, num_choices, dim=0)
                    list_broadcast_pastKeyValues_perLayer.append(broadcast_flatten_keyOrValue.flatten(0, 1))
                else:
                    raise ValueError(f"Invalid cached key or value shape: ", key_or_value.shape)

            list_broadcast_pastKeyValues.append(tuple(list_broadcast_pastKeyValues_perLayer))

        return input_masks, tuple(list_broadcast_pastKeyValues)

    def compute_allChoices_logProb_fromDecoderOutput(self,
                                                     input_masks,
                                                     past_key_values,
                                                     allChoices_ids,
                                                     allChoices_masks,
                                                     lengthNormalization):
        '''

        Args:
            input_masks: [batch_size, max_input_len]
            past_key_values: Tuple of keys and values for each layer.
                The first index of the tuple is the layer index, and the second index
                of the tuple is whether it is a key or value. Each element in tuple
                has shape [batch_size, max_input_len, num_heads, head_dim].
            allChoices_ids: [batch_size x num_choices, max_choice_len]
            allChoices_masks: [batch_size x num_choices, max_choice_len]

        Returns:
            logProbs_forAllChoices: [batch_size, num_choices]
            logProbs_forAllChoicesIds_zeroOutPadIds: [batch_size, num_choices, max_choice_len]
            len_allChoices: [batch_size ]
        '''
        num_choices = allChoices_ids.shape[0] // input_masks.shape[0]
        input_masks, past_key_values = self._broadcast_tensors(input_masks, past_key_values, num_choices)

        # Combine the input mask and choice mask so the model knows which cached input representations
        # are padded when conditioning on the cached input representations.
        # [batch_size x num_choices, max_input_len + max_choice_len]
        combined_mask = torch.cat([input_masks, allChoices_masks], dim=1)

        if self.use_position_ids:
            # Construct initial position ids solely based on choice lengths
            # [1, max_choice_len]
            allChoices_positionIds = torch.arange(0, allChoices_ids.shape[-1], dtype=torch.long, device=allChoices_ids.device)[None, :]
            input_len = torch.sum(input_masks, dim=1)[:, None]
            # Increment the position id to account for the input len.
            allChoices_positionIds = allChoices_positionIds + input_len

            # WARNING: The loss at transformer_outputs[0] is not valid, since allChoices_ids uses a
            # pad token of 0 and so the loss will not be ignored for the pad tokens
            transformer_outputs = self.transformer(input_ids=allChoices_ids,
                                                  attention_mask=combined_mask,
                                                  position_ids=allChoices_positionIds,
                                                  past_key_values=past_key_values,
                                                  use_cache=True,
                                                  labels=allChoices_ids)
        else:
            # WARNING: The loss at transformer_outputs[0] is not valid, since allChoices_ids uses a
            # pad token of 0 and so the loss will not be ignored for the pad tokens
            transformer_outputs = self.transformer(input_ids=allChoices_ids,
                                                  attention_mask=combined_mask,
                                                  past_key_values=past_key_values,
                                                  use_cache=True,
                                                  labels=allChoices_ids)


        # We used the logits for all choices to compute the log probs per example since
        # the loss returned in transformer_outputs will average the negative log probs across
        # examples
        # [batch_size x num_choices, max_choice_len, vocab_size]
        logits_ofAllChoices = transformer_outputs[1].float()

        # Shift the ids, masks, logits to handle predicting the next token for the decoder.
        # Note that we need to pass in the input_ids and cannot rely on HuggingFace automatically
        #   constructing the ids from the labels, since we need to pass in an attention mask to handle
        #   the cached input representations.
        shiftedLogits_ofAllChoices = logits_ofAllChoices[..., :-1, :].contiguous()
        shiftedIds_ofAllChoices = allChoices_ids[..., 1:].contiguous()
        shiftedMasks_ofAllChoices = allChoices_masks[..., 1:].contiguous()

        maxChoice_len = shiftedLogits_ofAllChoices.shape[1]
        vocab_size = shiftedLogits_ofAllChoices.shape[-1]

        # Compute the log probability of the ids for all choices with respect to the logits
        # [batch_size x num_choices x (max_choice_len-1)]
        logProbs_forAllChoices_ids = - F.cross_entropy(shiftedLogits_ofAllChoices.view(-1, vocab_size),
                                                 shiftedIds_ofAllChoices.view(-1),
                                                 reduction="none")

        return compute_logProb(logProbs_forAllChoices_ids,
                               shiftedMasks_ofAllChoices,
                               num_choices,
                               maxChoice_len,
                               lengthNormalization)

    def compute_allChoices_logProb_fromDecoderOutput_iteratively(self,
                                                                 input_masks,
                                                                 past_key_values,
                                                                 allChoices_ids,
                                                                 allChoices_masks,
                                                                 lengthNormalization):
        '''
        Args:
            input_masks: [batch_size, max_input_len]
            past_key_values: Tuple of keys and values for each layer.
                The first index of the tuple is the layer index, and the second index
                of the tuple is whether it is a key or value. Each element in tuple
                has shape [batch_size, max_input_len, num_heads, head_dim].
            allChoices_ids: [batch_size x num_choices, max_choice_len]
            allChoices_masks: [batch_size x num_choices, max_choice_len]
            lengthNormalization:
        Returns:
            logProbs_forAllChoices: [batch_size, num_choices]
            logProbs_forAllChoicesIds_zeroOutPadIds: [batch_size, max_choice_len, ]
            len_allChoices: [batch_size ]
        '''
        batch_size = input_masks.shape[0]
        assert batch_size == 1, "No need to score choices iteratively if batch size can be larger than 1"
        num_choices = allChoices_ids.shape[0] // input_masks.shape[0]

        list_logProbs_ofAllChoices = []
        list_logProbs_ofAllChoicesIds_zeroOutPadIds = []
        list_lenAllChoices = []

        for choice_idx in range(num_choices):
            # [1, max_choice_len]
            curChoice_ids = allChoices_ids[choice_idx:choice_idx + 1, :]
            curChoice_mask = allChoices_masks[choice_idx:choice_idx + 1, :]

            # Remove pad tokens
            assert  curChoice_mask.shape[0] == 1
            num_nonPadTokens = torch.sum(curChoice_mask)
            num_PadTokens = curChoice_mask.shape[1] - num_nonPadTokens

            curChoice_ids = curChoice_ids[:,:num_nonPadTokens]
            curChoice_mask = curChoice_mask[:,:num_nonPadTokens]

            assert curChoice_mask[0,-1] == 1

            # Combine the input mask and choice mask so the model knows which cached input representations
            # are padded when conditioning on the cached input representations.
            # [batch_size, max_input_len + max_choice_len]
            combined_mask = torch.cat([input_masks, curChoice_mask], dim=1)

            if self.use_position_ids:
                # Construct initial position ids solely based on choice lengths
                # [1, max_choice_len]
                curChoice_positionIds = torch.arange(0, curChoice_ids.shape[-1], dtype=torch.long,
                                                     device=curChoice_ids.device)[None, :]
                input_len = torch.sum(input_masks, dim=1)[:, None]
                # Increment the position id to account for the input len.
                curChoice_positionIds = curChoice_positionIds + input_len

                # WARNING: The loss at transformer_outputs[0] is not valid, since allChoices_ids uses a
                # pad token of 0 and so the loss will not be ignored for the pad tokens
                transformer_outputs = self.transformer(input_ids=curChoice_ids,
                                                       attention_mask=combined_mask,
                                                       position_ids=curChoice_positionIds,
                                                       past_key_values=past_key_values,
                                                       use_cache=True,
                                                       labels=curChoice_ids)
            else:
                # WARNING: The loss at transformer_outputs[0] is not valid, since allChoices_ids uses a
                # pad token of 0 and so the loss will not be ignored for the pad tokens
                transformer_outputs = self.transformer(input_ids=curChoice_ids,
                                                       attention_mask=combined_mask,
                                                       past_key_values=past_key_values,
                                                       use_cache=True,
                                                       labels=curChoice_ids)

            # We used the logits for all choices to compute the log probs per example since
            # the loss returned in transformer_outputs will average the negative log probs across
            # examples
            # [batch_size, max_choice_len, vocab_size]
            logits_ofCurChoice = transformer_outputs[1].float()

            # Shift the ids, masks, logits to handle predicting the next token for the decoder.
            # Note that we need to pass in the input_ids and cannot rely on HuggingFace automatically
            #   constructing the ids from the labels, since we need to pass in an attention mask to handle
            #   the cached input representations.
            shiftedLogits_ofCurChoice = logits_ofCurChoice[..., :-1, :].contiguous()
            shifted_curChoice_ids = curChoice_ids[..., 1:].contiguous()
            shifted_curChoice_mask = curChoice_mask[..., 1:].contiguous()

            maxChoice_len = shiftedLogits_ofCurChoice.shape[1]
            vocab_size = shiftedLogits_ofCurChoice.shape[-1]

            # Compute the log probability of the ids for all choices with respect to the logits
            # [batch_size x (max_choice_len-1)]
            logProbs_ofCurChoice_ids = - F.cross_entropy(shiftedLogits_ofCurChoice.view(-1, vocab_size),
                                                          shifted_curChoice_ids.view(-1),
                                                          reduction="none")

            # Compute the log probabilities of all the choices by averaging the log probabilities of
            # the ids and zeroing out the pad ids
            # [batch_size, (max_choice_len-1)]
            logProbs_ofCurChoice_ids = logProbs_ofCurChoice_ids.reshape(-1, maxChoice_len)
            shifted_curChoice_mask = shifted_curChoice_mask > 0
            logProbs_ofCurChoiceIds_zeroOutPadIds = logProbs_ofCurChoice_ids * shifted_curChoice_mask

            logProb_ofCurChoice = torch.sum(logProbs_ofCurChoiceIds_zeroOutPadIds, dim=1)
            len_curChoice = torch.sum(shifted_curChoice_mask, dim=1)

            if lengthNormalization:
                logProb_ofCurChoice = logProb_ofCurChoice / len_curChoice

            list_logProbs_ofAllChoices.append(logProb_ofCurChoice)
            list_logProbs_ofAllChoicesIds_zeroOutPadIds.append(torch.cat([
                                                                   logProbs_ofCurChoiceIds_zeroOutPadIds,
                                                                    torch.zeros((1, num_PadTokens)).to(logProbs_ofCurChoiceIds_zeroOutPadIds.device)
                                                               ], dim=1))
            list_lenAllChoices.append(len_curChoice)

        # Since batch size was 1, the batch size will be flattened and we have to add back the extra dimension with stack
        return torch.stack(list_logProbs_ofAllChoices, dim=1), \
               torch.stack(list_logProbs_ofAllChoicesIds_zeroOutPadIds, dim=1), \
               torch.stack(list_lenAllChoices, dim=1)

    def compute_allChoices_logProb(self,
                                   input_ids,
                                   input_masks,
                                   allChoices_ids,
                                   allChoices_masks,
                                   lengthNormalization,
                                   iterativelyComputeChoices):
        '''
        
        
        Args:
            input_ids: [batch_size, max_input_len]
            input_masks: [batch_size, max_input_len]
            allChoices_ids: [batch_size x num_choices, max_choice_len]
            allChoices_masks: [batch_size x num_choices, max_choice_len]
            lengthNormalization:
            iterativelyComputeChoices

        Returns:
            log_prob: [batch_size, num_choices]
        '''
        output = self.transformer(input_ids=input_ids, attention_mask=input_masks)
        past_key_values = output.past_key_values

        if iterativelyComputeChoices:
            return self.compute_allChoices_logProb_fromDecoderOutput_iteratively(input_masks,
                                                                                     past_key_values,
                                                                                     allChoices_ids,
                                                                                     allChoices_masks,
                                                                                     lengthNormalization)
        else:
            return self.compute_allChoices_logProb_fromDecoderOutput(input_masks,
                                                                         past_key_values,
                                                                         allChoices_ids,
                                                                         allChoices_masks,
                                                                         lengthNormalization)

    def predict_mulChoice(self, batch, pointMutualInfo, lengthNormalization, iterativelyComputeChoices):
        '''

        Args:
            batch:
            pointMutualInfo:
            lengthNormalization:

        Returns:
            pred_choice: [batch_size, ]
            score_ofChoices: [batch_size, num_choices]
            logProbs_ofAllChoicesIds: [batch_size, num_choices, max_choice_len]
            len_allChoices: [batch_size]
            logProbs_ofAllChoicesIds_condOnNullInput: [batch_size, num_choices, max_choice_len]
        '''
        # Compute log p(y|x)
        score_ofChoices, logProbs_ofAllChoicesIds, len_allChoices = self.compute_allChoices_logProb(
            batch["input_ids"],
            batch["input_masks"],
            batch["all_choices_ids"],
            batch["all_choices_masks"],
            lengthNormalization,
            iterativelyComputeChoices)

        logProbs_ofAllChoicesIds_condOnNullInput = None

        # For computing pointwise mutual information, we need to compute log p(y|x) - log p(y).
        # To compute p(y), we condition the choices on the null input.
        if pointMutualInfo:
            logProb_ofChoices_condOnNullInput, logProbs_ofAllChoicesIds_condOnNullInput, _ = self.compute_allChoices_logProb(
                batch["null_input_ids"],
                batch["null_input_masks"],
                batch["all_choices_ids"],
                batch["all_choices_masks"],
                lengthNormalization,
                iterativelyComputeChoices)
            score_ofChoices -= logProb_ofChoices_condOnNullInput

        _, pred_choice = torch.max(score_ofChoices, dim=1)
        return pred_choice.cpu().numpy().tolist(), \
               score_ofChoices.cpu().numpy().tolist(), \
               logProbs_ofAllChoicesIds.cpu().numpy().tolist(), \
               len_allChoices.cpu().numpy().tolist(), \
               logProbs_ofAllChoicesIds_condOnNullInput.cpu().numpy().tolist() if logProbs_ofAllChoicesIds_condOnNullInput is not None else None
