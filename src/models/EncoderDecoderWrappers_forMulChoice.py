import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils import compute_logProb


class EncoderDecoderWrappers_forMulChoice(nn.Module):
    '''

    '''

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def _broadcast_tensors(self, input_masks, encoder_outputs, num_choices):
        '''
        Broadcast the input masks and encoder outputs to account for multiple choices per input

        Args:
            input_masks: [batch_size, max_input_len]
            encoder_outputs: BaseModelOutput object from HuggingFace where the first element is
                            the hidden states of the encoder at the last layer
                            [batch_size, max_input_len, ff_dim]
            num_choices:

        Returns:
            input_masks: [batch_size x num_choices, max_input_len]
            encoder_outputs: BaseModelOutput object from HuggingFace where the first element is
                            the hidden states of the encoder at the last layer
                            [batch_size x num_choices, max_input_len, ff_dim]
        '''
        input_masks = torch.repeat_interleave(input_masks, num_choices, dim=0)
        encoder_outputs = (torch.repeat_interleave(encoder_outputs[0], num_choices, dim=0), )
        return input_masks, encoder_outputs

    def compute_allChoices_logProb_fromEncoderOutput(self,
                                                     input_masks,
                                                     encoder_outputs,
                                                     allChoices_ids,
                                                     allChoices_masks,
                                                     lengthNormalization):
        '''

        Args:
            input_masks: [batch_size, max_input_len]
            encoder_outputs: BaseModelOutput object from HuggingFace where the first element is
                            the hidden states of the encoder at the last layer
                            [batch_size, max_input_len, ff_dim]
            allChoices_ids: [batch_size x num_choices, max_choice_len]
            allChoices_masks: [batch_size x num_choices, max_choice_len]
            lengthNormalization:

        Returns:
            logProbs_forAllChoices: [batch_size, num_choices]
            logProbs_forAllChoicesIds_zeroOutPadIds: [batch_size, num_choices, max_choice_len]
        '''
        assert  allChoices_ids.shape[0] % input_masks.shape[0] == 0, \
            f"The batch size {allChoices_ids.shape[0]} of allChoices_ids is not a multiple of " \
            f"the batch size {input_masks.shape[0]} of input_masks"
        num_choices = allChoices_ids.shape[0] // input_masks.shape[0]

        input_masks, encoder_outputs = self._broadcast_tensors(input_masks, encoder_outputs, num_choices)

        # WARNING: The loss at transformer_outputs[0] is not valid, since allChoices_ids uses a
        # pad token of 0 and so the loss will not be ignored for the pad tokens
        # The input mask is passed in for the cross encoder-decoder attention.
        transformer_outputs = self.transformer(attention_mask=input_masks,
                                      encoder_outputs=encoder_outputs,
                                      labels=allChoices_ids)

        # We used the logits for all choices to compute the log probs per example since
        # the loss returned in transformer_outputs will average the negative log probs across
        # examples
        # [batch_size x num_choices, max_choice_len, vocab_size]
        logits_ofAllChoices = transformer_outputs[1].float()
        maxChoice_len = logits_ofAllChoices.shape[1]
        vocab_size = logits_ofAllChoices.shape[-1]

        # Compute the log probability of the ids for all choices with respect to the logits
        # [batch_size x num_choices x max_choice_len]
        logProbs_ofAllChoices_ids = - F.cross_entropy(logits_ofAllChoices.view(-1, vocab_size),
                                                 allChoices_ids.view(-1),
                                                 reduction="none")

        return compute_logProb(logProbs_ofAllChoices_ids,
                               allChoices_masks,
                               num_choices,
                               maxChoice_len,
                               lengthNormalization)


    def compute_allChoices_logProb(self,
                                   input_ids,
                                   input_masks,
                                   allChoices_ids,
                                   allChoices_masks,
                                   lengthNormalization):
        '''
        
        
        Args:
            input_ids: [batch_size, max_input_len]
            input_masks: [batch_size, max_input_len]
            allChoices_ids: [batch_size x num_choices, max_choice_len]
            allChoices_masks: [batch_size x num_choices, max_choice_len]
            lengthNormalization:

        Returns:
            log_prob: [batch_size x num_choices, max_choice_len]
        '''
        # Search for encoder function
        if hasattr(self.transformer, "encoder"):
            encoder_outputs = self.transformer.encoder(input_ids, input_masks)
        elif hasattr(self.transformer, "model") and hasattr(self.transformer.model, "encoder"):
            encoder_outputs = self.transformer.model.encoder(input_ids, input_masks)
        else:
            raise ValueError("Cannot find encoder function in transformer")

        return self.compute_allChoices_logProb_fromEncoderOutput(input_masks,
                                                                 encoder_outputs,
                                                                 allChoices_ids,
                                                                 allChoices_masks,
                                                                 lengthNormalization)

    def predict_mulChoice(self, batch, pointMutualInfo, lengthNormalization, iterativelyComputeChoices):
        '''

        Args:
            batch:
            pointMutualInfo:
            lengthNormalization:
            iterativelyComputeChoices: Not used. Added to be consistent with DecoderWrappers_forMulChoice

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
            lengthNormalization)

        logProbs_ofAllChoicesIds_condOnNullInput = None

        # For computing pointwise mutual information, we need to compute log p(y|x) - log p(y).
        # To compute p(y), we condition the choices on the null input.
        if pointMutualInfo:
            logProb_ofChoices_condOnNullInput, logProbs_ofAllChoicesIds_condOnNullInput, _ = self.compute_allChoices_logProb(
                batch["null_input_ids"],
                batch["null_input_masks"],
                batch["all_choices_ids"],
                batch["all_choices_masks"],
                lengthNormalization)
            score_ofChoices -= logProb_ofChoices_condOnNullInput

        _, pred_choice = torch.max(score_ofChoices, dim=1)

        return pred_choice.cpu().numpy().tolist(), \
               score_ofChoices.cpu().numpy().tolist(), \
               logProbs_ofAllChoicesIds.cpu().numpy().tolist(), \
               len_allChoices.cpu().numpy().tolist(), \
               logProbs_ofAllChoicesIds_condOnNullInput.cpu().numpy().tolist() if logProbs_ofAllChoicesIds_condOnNullInput is not None else None
