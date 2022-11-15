import json
import argparse
import torch
import tqdm
import os
import logging
import torch.distributed as dist

import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig

from src.data.multiple_choice import MultipleChoiceDataset, MultipleChoiceReader
from src.data.Batcher import Batcher


from src.eval.Scorer import Scorer
from src.eval.PredictionLogger import PredictionLogger

from src.constructors import construct_hugFace_objects, construct_models

from src.utils.util import set_seeds, reduce_gatheredOutput, get_mulChoice_outputDir
from src.utils.deepspeed import get_deepspeedConfig

def evaluate_dataset(mul_choice_filepath,
                     deepspeed_engine,
                     model,
                     world_size,
                     eval_batchSize,
                     device,
                     ignore_pointwise_mutual_information,
                     ignore_length_normalization,
                     compute_choices_iteratively,
                     model_name,
                     tokenizer,
                     prompt_template_idx,
                     input_prefix,
                     debug):

    mcReader = MultipleChoiceReader()
    createDataset_fn = lambda data: MultipleChoiceDataset(data, tokenizer, prompt_template_idx, input_prefix, device, world_size)
    batcher = Batcher(mcReader, createDataset_fn, train_batchSize=None, eval_batchSize=eval_batchSize)

    if world_size is None or dist.get_rank() == 0:
        scorer = Scorer("multiple_choice")
        output_dir = get_mulChoice_outputDir(mul_choice_filepath, model_name, ignore_pointwise_mutual_information, ignore_length_normalization)
        if debug:
            output_fp = os.path.join("exp_out", "multiple_choice", "debug.json")
        else:
            output_fp = os.path.join(output_dir, f"predictions-prompt_{prompt_template_idx}.json")
            if os.path.exists(output_fp):
                return
        prediction_logger = PredictionLogger(output_fp)


    for batch in tqdm.tqdm(batcher.get_mulChoiceBatches(mul_choice_filepath)):
        with torch.no_grad():
            # Uses deepspeed
            if world_size is not None:
                pred_choice, score_ofChoices, logProbs_ofAllChoicesIds, len_allChoices, logProbs_ofAllChoicesIds_condOnNullInput = deepspeed_engine.module.predict_mulChoice(batch,
                                                                                            not ignore_pointwise_mutual_information,
                                                                                            not ignore_length_normalization,
                                                                                            compute_choices_iteratively)
            else:
                pred_choice, score_ofChoices, logProbs_ofAllChoicesIds, len_allChoices, logProbs_ofAllChoicesIds_condOnNullInput = model.predict_mulChoice(batch,
                                                                          not ignore_pointwise_mutual_information,
                                                                          not ignore_length_normalization,
                                                                          compute_choices_iteratively)

            batchOf_evalInfo = {
                "pred_choice": pred_choice,
                "score_of_choices": score_ofChoices,
                "log_probs_of_all_choices_ids": logProbs_ofAllChoicesIds,
                "len_all_choices": len_allChoices,
                "log_prob_of_all_choices_ids_cond_null_input": logProbs_ofAllChoicesIds_condOnNullInput if logProbs_ofAllChoicesIds_condOnNullInput is not None else [0 * len(logProbs_ofAllChoicesIds)],
                "input": batch["input"],
                "list_choices": batch["list_choices"],
                "lbl": batch["lbl"].cpu().numpy().tolist()
            }

            if world_size is not None:
                listOf_batchOf_evalInfo = [{}] * world_size
                dist.gather_object(
                    batchOf_evalInfo,
                    listOf_batchOf_evalInfo if dist.get_rank() == 0 else None,
                    dst=0
                )
                if dist.get_rank() == 0:
                    batchOf_evalInfo = reduce_gatheredOutput(listOf_batchOf_evalInfo)

            if world_size is None or dist.get_rank() == 0:
                whichDatapoints_correct = scorer.add_batch(batchOf_evalInfo)
                batchOf_evalInfo.update({
                    "is_datapoint_correct": whichDatapoints_correct
                })
                prediction_logger.log_batch(batchOf_evalInfo)

    if not debug:
        if world_size is None or dist.get_rank() == 0:
            with open(os.path.join(output_dir, "scores.jsonl"), 'a+') as f_out:
                dict_score = scorer.get_score()
                dict_score.update({
                    "pointwise_mutual_information": not ignore_pointwise_mutual_information,
                    "length_normalization": not ignore_length_normalization,
                    "dataset_filepath": mul_choice_filepath,
                    "model": model_name,
                    "prompt_template_idx": prompt_template_idx
                })
                f_out.write(json.dumps(dict_score) + '\n')


def evaluate_mulChoice(args):

    # Uses deepspeed
    if args.world_size is not None:
        hugFace_config, tokenizer, input_prefix = construct_hugFace_objects(args.model_name, args.max_seq_len)
        if hasattr(hugFace_config, "d_model"):
            model_dim = hugFace_config.d_model
        elif hasattr(hugFace_config, "hidden_size"):
            model_dim = hugFace_config.hidden_size
        else:
            raise ValueError("Cannot get model dimension from hugging face config")

        deepspeed_config = get_deepspeedConfig(args.eval_batch_size, args.world_size, model_dim)

        model, _ = construct_models(args.model_name, args.use_hugFace_parallelism, args.use_bitsandbytes)
        dschf = HfDeepSpeedConfig(deepspeed_config)  # keep this object alive and create it before initializing the model

        deepspeed_engine = deepspeed.init_inference(model,
                                          mp_size=args.world_size,
                                          dtype=torch.float,
                                          replace_method='auto',
                                          replace_with_kernel_inject=True)
        deepspeed_engine.module.eval()  # inference
        model = None
    else:
        hugFace_config, tokenizer, input_prefix = construct_hugFace_objects(args.model_name, args.max_seq_len)
        model, _ = construct_models(args.model_name, args.use_hugFace_parallelism, args.use_bitsandbytes)

        if not args.use_hugFace_parallelism and not args.use_bitsandbytes:
            model = model.to(args.device)

        model.eval()
        deepspeed_engine = None

    for mul_choice_filepath in args.mul_choice_filepath:
        if args.prompt_template_idx == -1:
            for prompt_template_idx in range(3):
                evaluate_dataset(mul_choice_filepath,
                                 deepspeed_engine,
                                 model,
                                 args.world_size,
                                 args.eval_batch_size,
                                 args.device,
                                 args.ignore_pointwise_mutual_information,
                                 args.ignore_length_normalization,
                                 args.compute_choices_iteratively,
                                 args.model_name,
                                 tokenizer,
                                 prompt_template_idx,
                                 input_prefix,
                                 args.debug)
        else:
            evaluate_dataset(mul_choice_filepath,
                             deepspeed_engine,
                             model,
                             args.world_size,
                             args.eval_batch_size,
                             args.device,
                             args.ignore_pointwise_mutual_information,
                             args.ignore_length_normalization,
                             args.compute_choices_iteratively,
                             args.model_name,
                             tokenizer,
                             args.prompt_template_idx,
                             input_prefix,
                             args.debug)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--mul_choice_filepath", action='store', type=str, nargs='*', required=True)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument('-m', "--model_name", required=True)
    parser.add_argument("--use_deepspeed", action="store_true")
    parser.add_argument("--use_bitsandbytes", action="store_true")
    parser.add_argument("--use_hugFace_parallelism", action="store_true")
    parser.add_argument('-b', "--eval_batch_size", type=int, default=1)
    parser.add_argument('-p', "--prompt_template_idx", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--ignore_pointwise_mutual_information',
                        action="store_true",
                        help="Whether to use the pointwise mutual information or regular log "
                             "likelihood for scoring candidates")
    parser.add_argument('--ignore_length_normalization',
                        action="store_true",
                        help="Whether to use the whether to use length normalization when scoring the candidates ")
    parser.add_argument('--compute_choices_iteratively',
                        action="store_true",
                        help="Whether to use compute log probs of decoder choices together or iteratively")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info('Starting evaluate multiple choice')

    if args.use_deepspeed:
        logging.info('Using Deepspeed')
        # The device is the local_rank since it specifies the GPU to use.
        args.device = args.local_rank
        args.world_size = int(os.getenv('WORLD_SIZE', '1'))
        deepspeed.init_distributed()
    else:
        # This device is where the input_ids will be loaded.
        # It must be 0 since using huggingface parallelism assumes the logits should be back on device 0 to compute the
        # loss with the input_ids
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.world_size = None

    evaluate_mulChoice(args)