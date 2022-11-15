import argparse
import json
import statistics
import os
import numpy as np

from src.get_results import get_medianScore_perDataset



def get_fibResult(model, datasets, dataset_len):


    acc_acrossDataset = []
    for dataset in datasets:
        acc_perDataset = get_medianScore_perDataset(dataset, [model])
        acc_acrossDataset.append(acc_perDataset[0])


    acc_acrossDataset = np.asarray(acc_acrossDataset) * 100
    final_score = float(np.dot(acc_acrossDataset, dataset_len) / np.sum(dataset_len))
    print(f"The final score is {round(final_score, 3)}")

def getXSum_fibResults(model):

    datasets = ["exp_out/multiple_choice/xsum/fib/binary_choice-using_bart-base_distractors",
                "exp_out/multiple_choice/xsum/fib/binary_choice-using_bart-large_distractors",
                "exp_out/multiple_choice/xsum/fib/binary_choice-using_bloom-560m_distractors",
                "exp_out/multiple_choice/xsum/fib/binary_choice-using_distil-bart_distractors",
                "exp_out/multiple_choice/xsum/fib/binary_choice-using_distil-pegasus_distractors",
                "exp_out/multiple_choice/xsum/fib/binary_choice-using_pegasus_distractors",
                "exp_out/multiple_choice/xsum/fib/binary_choice-using_t5-large_distractors"]

    dataset_len = np.asarray([463, 414, 479, 410, 437, 438, 483])

    get_fibResult(model, datasets, dataset_len)


def getCNNDM_fibResults(model):

    datasets = ["exp_out/multiple_choice/cnn_dm/fib/binary_choice-using_banditsumm_distractors",
                "exp_out/multiple_choice/cnn_dm/fib/binary_choice-using_bert_lstm_pn_rl_distractors",
                "exp_out/multiple_choice/cnn_dm/fib/binary_choice-using_heter_graph_distractors",
                "exp_out/multiple_choice/cnn_dm/fib/binary_choice-using_lead3_distractors",
                "exp_out/multiple_choice/cnn_dm/fib/binary_choice-using_matchsumm_distractors",
                "exp_out/multiple_choice/cnn_dm/fib/binary_choice-using_mi_unsup_distractors",
                "exp_out/multiple_choice/cnn_dm/fib/binary_choice-using_neusumm_distractors",
                "exp_out/multiple_choice/cnn_dm/fib/binary_choice-using_oracle_disco_distractors",
                "exp_out/multiple_choice/cnn_dm/fib/binary_choice-using_oracle_distractors",
                "exp_out/multiple_choice/cnn_dm/fib/binary_choice-using_pacsum_bert_distractors",
                "exp_out/multiple_choice/cnn_dm/fib/binary_choice-using_pacsum_tfidf_distractors",
                "exp_out/multiple_choice/cnn_dm/fib/binary_choice-using_refresh_distractors",
                "exp_out/multiple_choice/cnn_dm/fib/binary_choice-using_rnn_ext_rl_distractors",
                "exp_out/multiple_choice/cnn_dm/fib/binary_choice-using_textrank_distractors",
                "exp_out/multiple_choice/cnn_dm/fib/binary_choice-using_textrank_st_distractors"]

    dataset_len = np.asarray([26, 23, 22, 5, 21, 34, 24, 72, 54, 12, 27, 31, 24, 36, 46])

    get_fibResult(model, datasets, dataset_len)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model", type=str, required=True)
    parser.add_argument('-d', "--dataset", choices=["xsum", "cnn_dm"])
    args = parser.parse_args()

    if args.dataset == "xsum":
        getXSum_fibResults(args.model)
    else:
        getCNNDM_fibResults(args.model)