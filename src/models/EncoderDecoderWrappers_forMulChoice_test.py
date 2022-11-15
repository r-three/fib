import torch

from src.data.multiple_choice import MultipleChoiceDataset, MultipleChoiceReader
from src.data.Batcher import Batcher

from src.constructors import construct_hugFace_objects, construct_models

from src.models.EncoderDecoderWrappers_forMulChoice import EncoderDecoderWrappers_forMulChoice

class Test_EncoderDecoderWrappers(EncoderDecoderWrappers_forMulChoice):
    '''

    '''

    def __init__(self, transformer):
        super().__init__(transformer)
        self.transformer = transformer

    def _test_broadcast_tensor(self, input_mask, encoder_outputs, num_choices):
        '''
        Test that when repeating the tensors by num_choices times, the repetitions will  
        be in the same block. 
        
        Args:
            input_masks: [batch_size, max_input_len]
            encoder_outputs: BaseModelOutput object from HuggingFace where the first element is
                            the hidden states of the encoder at the last layer
                            [batch_size, max_input_len, ff_dim]
            num_choices:
        '''
        new_inputMask, new_encoderOutputs = \
            super()._broadcast_tensors(input_mask, encoder_outputs, num_choices)
        
        batch_size = input_mask.shape[0]
        for i in range(batch_size):
            assert torch.equal(input_mask[i:i+1].repeat(num_choices, 1) ,
                               new_inputMask[i*num_choices:(i+1)*num_choices, ]), \
                   f"Test of broadcasting input_mask failed."
            assert torch.equal(encoder_outputs[0][i:i+1].repeat(num_choices, 1, 1),
                               new_encoderOutputs[0][i*num_choices:(i+1)*num_choices]), \
                   f"Test of broadcasting encoder_outputs failed."
            


    def test_compute_allChoices_logProb_fromEncoderOutput(self,
                                                          input_ids,
                                                          input_masks,
                                                          encoder_outputs,
                                                          allChoices_ids,
                                                          allChoices_masks,
                                                          allChoices_lbls):
        '''

        Args:
            input_ids: [batch_size, max_input_len]
            input_masks: [batch_size, max_input_len]
            encoder_outputs: BaseModelOutput object from HuggingFace where the first element is
                            the hidden states of the encoder at the last layer
                            [batch_size, max_input_len, ff_dim]
            allChoices_ids: [batch_size x num_choices, max_choice_len]
            allChoices_masks: [batch_size x num_choices, max_choice_len]
            allChoices_lbls: [batch_size x num_choices, max_choice_len]

        Returns:

        '''
        num_choices = allChoices_lbls.shape[0] // input_masks.shape[0]
        batch_size = input_masks.shape[0]
        self._test_broadcast_tensor(input_masks, encoder_outputs, num_choices)

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
                transformer_outputs = self.transformer(input_ids=datapoint_ids,
                                                      attention_mask=datapoint_mask,
                                                      labels=choice_lbls,
                                                       output_hidden_states=True)
                choice_logProb = - transformer_outputs[0]
                listOf_logProb.append(choice_logProb)

        logProb_forAllChoices = torch.stack(listOf_logProb, dim=0).reshape(batch_size, num_choices)

        assert torch.allclose(logProb_forAllChoices,
                              super().compute_allChoices_logProb_fromEncoderOutput(
                                                input_masks,
                                                encoder_outputs,
                                                allChoices_ids,
                                                allChoices_masks,
                                                True)[0],
                              atol=1e-4), \
                "Test of computing log probs from encoder output failed."

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
        # Search for encoder function
        if hasattr(self.transformer, "encoder"):
            encoder_outputs = self.transformer.encoder(input_ids, input_masks)
        elif hasattr(self.transformer, "model") and hasattr(self.transformer.model, "encoder"):
            encoder_outputs = self.transformer.model.encoder(input_ids, input_masks)
        else:
            raise ValueError("Cannot find encoder function in transformer")

        self.test_compute_allChoices_logProb_fromEncoderOutput(input_ids,
                                                               input_masks,
                                                               encoder_outputs,
                                                               allChoices_ids,
                                                               allChoices_masks,
                                                               allChoices_lbls)

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
    for model_name in ["Frederick0291/t5-small-finetuned-xsum", "facebook/bart-large-xsum"]:
        hugFace_config, tokenizer, input_prefix = construct_hugFace_objects(model_name, 512)
        _, transformer = construct_models(model_name, False, False)

        model = Test_EncoderDecoderWrappers(transformer).to(device)
        model.eval()

        mcReader = MultipleChoiceReader()
        createDataset_fn = lambda data: MultipleChoiceDataset(data, tokenizer, 0, input_prefix, device, world_size=None)
        batcher = Batcher(mcReader, createDataset_fn, train_batchSize=None, eval_batchSize=2)

        for i, batch in enumerate(batcher.get_mulChoiceBatches("multiple_choice-dataset/xsum/random_distractors/binary_choice-using_random_distractors.jsonl")):
            with torch.no_grad():
                model.test_predict_mulChoice(batch)
            if i > 4:
                break