

def get_deepspeedConfig(eval_batchSize, world_size, model_dim):
    '''

    Args:
        eval_batchSize:
        world_size:
        model_dim:

    Returns:

    '''
    # https://github.com/huggingface/transformers/issues/15399
    deepspeed_config = {
        "fp16": {
            "enabled": False,
        },
        "bf16": {
            "enabled": False,
        },
        "zero_optimization": {
            "stage": 3,
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": model_dim * model_dim,
            "stage3_prefetch_bucket_size": 0.9 * model_dim * model_dim,
            "stage3_param_persistence_threshold": 10 * model_dim
        },
        "steps_per_print": 2000,
        "train_batch_size": eval_batchSize * world_size,
        "train_micro_batch_size_per_gpu": eval_batchSize,
        "wall_clock_breakdown": False
    }

    return deepspeed_config