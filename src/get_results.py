import argparse
import json
import statistics
import os


def get_medianScore_perExperiment(modelWithScoringFunction_dir):
    score_fp = os.path.join(modelWithScoringFunction_dir, "scores.jsonl")

    dict_promptTemplateIdx_toMCAccuracy = {}

    with open(score_fp, "r") as f:
        for line in f.readlines():
            score_json = json.loads(line.strip("\n"))
            dict_promptTemplateIdx_toMCAccuracy[score_json["prompt_template_idx"]] = score_json["multiple-choice-accuracy"]

    return statistics.median(list(dict_promptTemplateIdx_toMCAccuracy.values()))

def get_medianScore_perModel(model_dir):
    avg_pmi_acc = get_medianScore_perExperiment(model_dir)
    return [avg_pmi_acc]

def get_medianScore_perDataset(dataset_dir, list_models):
    list_acc = []

    for model in list_models:
        if model is None:
            list_acc.extend([0] * 4)
        else:
            model_dir = os.path.join(dataset_dir, model)
            list_acc.extend(get_medianScore_perModel(model_dir))

    return list_acc

def get_medianScore_acrossDatasets(datasets, list_models):

    print("Using the following datasets ... ")
    acc_acrossDataset = []
    for dataset in datasets:
        print(dataset)
        acc_perDataset = get_medianScore_perDataset(dataset, list_models)
        acc_acrossDataset.append(acc_perDataset)

    print("The median accuracy per model across different distractor models is ... ")
    for idx, acc_perModel in enumerate(list(map(list, zip(*acc_acrossDataset)))):
        formattedAcc_perModel = list(map(lambda x: str(round(x * 100, 3)), acc_perModel))
        print(list_models[idx] + ": " + ",".join(formattedAcc_perModel))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--exp_dir_of_datasets", action='store', type=str, nargs='*', required=True)
    parser.add_argument('-m', "--list_models", action='store', type=str, nargs='*', required=True)
    args = parser.parse_args()

    get_medianScore_acrossDatasets(args.exp_dir_of_datasets, args.list_models)