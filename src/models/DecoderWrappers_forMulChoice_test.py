import torch

from src.data.multiple_choice import MultipleChoiceDataset, MultipleChoiceReader
from src.data.Batcher import Batcher

from src.constructors import construct_hugFace_objects, construct_models
from src.models.DecoderWrappers_forMulChoice import DecoderWrappers_forMulChoice

class Test_DecoderWrappers(DecoderWrappers_forMulChoice):
    '''

    '''

    def __init__(self, transformer):
        super().__init__(transformer)
        self.transformer = transformer

    def _test_broadcast_tensor(self, input_masks, past_key_values, num_choices):
        '''
        Test that when repeating the tensors by num_choices times, the repetitions will  
        be in the same block. 
        
        Args:
            input_masks: [batch_size, max_input_len]
            past_key_values: Tuple of keys and values for each layer.
                The first index of the tuple is the layer index, and the second index
                of the tuple is whether it is a key or value. Each element in tuple
                has shape [batch_size, max_input_len, num_heads, head_dim] or [batch_size x num_heads, head_dim, max_input_len].
            num_choices:
        '''
        new_inputMask, new_pastKeyValues = super()._broadcast_tensors(input_masks, past_key_values, num_choices)

        batch_size = input_masks.shape[0]
        for i in range(batch_size):
            assert torch.equal(input_masks[i:i+1].repeat(num_choices, 1) ,
                               new_inputMask[i*num_choices:(i+1)*num_choices, ]), \
                   f"Test of broadcasting input_masks failed."
        for old_keyValues_perLayer, new_keyValues_perLayer in zip(past_key_values, new_pastKeyValues):
            for old_keyOrValue, new_keyOrValue in zip(old_keyValues_perLayer, new_keyValues_perLayer):
                batch_size = old_keyOrValue.shape[0]
                for i in range(batch_size):
                    num_heads = past_key_values[0][0].shape[0] // input_masks.shape[0]

                    # This means the keys or values are of shape [batch_size, max_input_len, num_heads, head_dim]
                    if num_heads == 1:
                        assert torch.equal(old_keyOrValue[i:i + 1].repeat(num_choices, 1, 1, 1),
                                           new_keyOrValue[i * num_choices:(i + 1) * num_choices]), \
                                f"Test of broadcasting key,values failed."
                    # This means the keys and values are of shape [batch_size x num_heads, head_dim, max_input_len].
                    else:
                        assert torch.equal(old_keyOrValue[i * num_heads : (i + 1) * num_heads].repeat((num_choices, 1, 1)),
                                       new_keyOrValue[i * num_heads * num_choices : (i + 1) * num_heads * num_choices]), \
                                        f"Test of broadcasting key,values failed."


    def test_compute_allChoices_logProb_fromDecoderOutput(self,
                                                          input_ids,
                                                          input_masks,
                                                          past_key_values,
                                                          allChoices_ids,
                                                          allChoices_masks,
                                                          allChoices_lbls):
        '''

        Args:
            input_ids: [batch_size, max_input_len]
            input_masks: [batch_size, max_input_len]
            past_key_values: Tuple of keys and values for each layer.
                The first index of the tuple is the layer index, and the second index
                of the tuple is whether it is a key or value. Each element in tuple
                has shape [batch_size, max_input_len, num_heads, head_dim] or [batch_size x num_heads, head_dim, max_input_len].
            allChoices_ids: [batch_size x num_choices, max_choice_len]
            allChoices_masks: [batch_size x num_choices, max_choice_len]
            allChoices_lbls: [batch_size x num_choices, max_choice_len]

        Returns:

        '''
        num_choices = allChoices_lbls.shape[0] // input_masks.shape[0]
        batch_size = input_masks.shape[0]
        self._test_broadcast_tensor(input_masks, past_key_values, num_choices)

        # Iterate over every datapoint and every choice to compute the log prob using the loss
        # returned from HuggingFace.  Since HuggingFace averages the loss per batch,
        # we use batch_size=1 to get the log prob for each choice of each datapoint.
        listOf_logProb = []
        for datapoint_idx in range(batch_size):
            datapoint_ids = input_ids[datapoint_idx:datapoint_idx + 1]
            datapoint_mask = input_masks[datapoint_idx:datapoint_idx + 1]

            for choice_idx in range(num_choices):
                choiceLbls_idx = datapoint_idx*num_choices + choice_idx
                choice_lbls = allChoices_lbls[choiceLbls_idx:choiceLbls_idx+1]
                choice_ids = allChoices_ids[choiceLbls_idx:choiceLbls_idx+1]
                choice_mask = allChoices_masks[choiceLbls_idx:choiceLbls_idx+1]

                # Note the batch size is 1. Have to filter the datapoint_ids
                # to remove the pad ids in between the datapoint and the choice when we combined them.
                datapoint_len = torch.sum(datapoint_mask)
                filtered_datapointIds = datapoint_ids[:,:datapoint_len]
                combined_ids = torch.cat([filtered_datapointIds, choice_ids], dim=1)
                combined_mask = torch.cat([datapoint_mask[:,:datapoint_len], choice_mask], dim=1)

                # We want to ignore the loss for the datapoint and only compute the loss for the choices.
                datapoint_lbls = torch.ones_like(filtered_datapointIds).to(datapoint_ids.device) * -100
                # We ignore the first token in choice labels since HuggingFace will shift the labels
                #   one over to the left, but since we concatenate the datapoint labels and choice labels,
                #   the first choice id will not be shifted over.
                choice_lbls[:,0] = -100
                combined_lbls = torch.cat([datapoint_lbls, choice_lbls], dim=1)
                transformer_outputs = self.transformer(input_ids=combined_ids,
                                                      attention_mask=combined_mask,
                                                      labels=combined_lbls)
                choice_logProb = - transformer_outputs[0]
                listOf_logProb.append(choice_logProb)

        logProb_forAllChoices = torch.stack(listOf_logProb, dim=0).reshape(batch_size, num_choices)

        assert torch.allclose(logProb_forAllChoices,
                              super().compute_allChoices_logProb_fromDecoderOutput(
                                                input_masks,
                                                past_key_values,
                                                allChoices_ids,
                                                allChoices_masks,
                                                True)[0],
                              atol=1e-4), \
                "Test of computing log probs from decoder output failed."

    def test_compute_allChoices_logProb_fromDecoderOutput_iteratively(self,
                                                          input_masks,
                                                          past_key_values,
                                                          allChoices_ids,
                                                          allChoices_masks):
        '''
        Args:
            input_masks: [batch_size, max_input_len]
            past_key_values: Tuple of keys and values for each layer.
                The first index of the tuple is the layer index, and the second index
                of the tuple is whether it is a key or value. Each element in tuple
                has shape [batch_size, max_input_len, num_heads, head_dim] or [batch_size x num_heads, head_dim, max_input_len].
            allChoices_ids: [batch_size x num_choices, max_choice_len]
            allChoices_masks: [batch_size x num_choices, max_choice_len]
        Returns:
        '''
        assert torch.allclose(super().compute_allChoices_logProb_fromDecoderOutput_iteratively(
                                  input_masks,
                                  past_key_values,
                                  allChoices_ids,
                                  allChoices_masks,
                                  True)[0],
                              super().compute_allChoices_logProb_fromDecoderOutput(
                                  input_masks,
                                  past_key_values,
                                  allChoices_ids,
                                  allChoices_masks,
                                  True)[0],
                              atol=1e-4), \
            "Test of computing log probs from decoder output failed."
    def test_compute_allChoices_logProb(self,
                                        input_ids,
                                        input_masks,
                                        allChoices_ids,
                                        allChoices_masks,
                                        allChoices_lbls):
        '''
        
        
        Args:
            input_ids: [batch_size, max_input_len]
            input_masks: [batch_size, max_input_len]
            allChoices_ids: [batch_size x num_choices, max_choice_len]
            allChoices_lbls: [batch_size x num_choices, max_choice_len]

        Returns:
            log_prob: [batch_size x num_choices, max_choice_len]
        '''
        output = self.transformer(input_ids=input_ids, attention_mask=input_masks)
        past_key_values = output.past_key_values

        self.test_compute_allChoices_logProb_fromDecoderOutput(input_ids,
                                                               input_masks,
                                                               past_key_values,
                                                               allChoices_ids,
                                                               allChoices_masks,
                                                               allChoices_lbls)

        self.test_compute_allChoices_logProb_fromDecoderOutput_iteratively(input_masks,
                                                                           past_key_values,
                                                                           allChoices_ids,
                                                                           allChoices_masks)

    def test_predict_mulChoice(self, batch):
        '''

        Args:
            batch:
            pointMutualInfo:

        Returns:
            predChoice: [batch_size, ]
            predProb: [batch_size, ]
        '''

        # Compute log p(y|x)
        self.test_compute_allChoices_logProb(
            batch["input_ids"],
            batch["input_masks"],
            batch["all_choices_ids"],
            batch["all_choices_masks"],
            batch["all_choices_lbls"])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # This only tests all models for 6 batches.
    for model_name in ["bigscience/bloom-560m", "gpt2", "facebook/opt-125m"]:
        hugFace_config, tokenizer, input_prefix = construct_hugFace_objects(model_name, 512)
        _, transformer = construct_models(model_name, False, False)

        model = Test_DecoderWrappers(transformer).to(device)
        model.eval()

        mcReader = MultipleChoiceReader()
        createDataset_fn = lambda data: MultipleChoiceDataset(data, tokenizer, 0, input_prefix, device, world_size=None)
        batcher = Batcher(mcReader, createDataset_fn, train_batchSize=None, eval_batchSize=1)

        for i, batch in enumerate(batcher.get_mulChoiceBatches("multiple_choice-dataset/xsum/random_distractors/binary_choice-using_random_distractors.jsonl")):
            with torch.no_grad():
                model.test_predict_mulChoice(batch)
            if i > 4:
                break