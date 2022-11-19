# FIB

## FIB Benchmark

The dataset can be found [here]() .
Note that the multiple-choice accuracy is computed slightly different way in our work. See [below](#evaluating-models-on-fIB) for more details. 



## Evaluating Models 

### Setup

1. Create a virtual environment and activate it.
```
python3 -m venv env
source env/bin/activate
```
2. Install dependencies 
```
python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
3. Set environment variables (This step has to be done every session.)
```
source bin/setup.sh
```

### Evaluating Models

The following command is used to evaluate models: 
```
python src/evaluate_mulChoice.py -f {multiple_choice-dataset_filepath} -m {model}
```
 
For example,
```commandline
python src/evaluate_mulChoice.py -f multiple_choice-dataset/xsum/fib/binary_choice-using_bart-base_distractors.jsonl -m facebook/opt-1.3b
```
Our code has only been tested on evaluating models from the BLOOM, OPT, GPT, and T5 families. 

Note that though DeepSpeed is implemented, we did not use it. So our implementation of DeepSpeed might have some bugs.

### Get Results 
The following command is used to gather multiple results and get the median score:
```
python src/scripts/get_results.py -e {all_experiment_directories_of_datasets} -m {list_models}
``` 
Note that``scores.json`` files contain the scores for all the prompts.

For example, 
```
python src/scripts/get_results.py -f exp_out/multiple_choice/xsum/fib/* -m bigscience-T0_3B
```

## Evaluating Models on FIB

The difference between the FIB dataset released above and the evaluation here is 
- Here, we take the median accuracy across of the model across 3 prompts for each distractor model used. Then, we take a weighted average of the median accuracies across different distractor models.
- In the FIB dataset, we combine all the examples from each distractor model and across XSum and CNN/DM into one file to simplify it. Users can use any prompt they want.

The following commands will run it. 
```
python src/evaluate_mulChoice.py -f multiple_choice-dataset/{dataset}/fib/binary_choice-* -m {model}
python src/compute_fib_results.py -m {model} -d {dataset}
```



## Other Binary Multiple-Choice Datasets

The datasets are under ``multiple_choice-dataset/xsum`` and ``multiple_choice-dataset/cnn_dm`` for XSum and CNN\DM respectively. 

The different alternative choices include
1. FIB - Our benchmark of factually inconsistent model-generated summaries
2. [FactCC](https://github.com/salesforce/factCC.git) 
3. [MFMA](https://github.com/hwanheelee1993/MFMA)
4. FIR - factually inconsistent reference summaries (i.e. reference summaries from XSum or CNN\DM that were annotated as factually inconsistent)
5. factually consistent model generated summaries. 

## Contact ##

For any doubts or questions regarding the work, please contact Derek ([dtredsox@cs.unc.edu](mailto:dtredsox+adapet@cs.unc.edu)). For any bug or issues with the code, feel free to open a GitHub issue or pull request.

## Citation ##


If you find this repo helpful, welcome to cite our work:

```

```

We use the following code in our works:

```
@inproceedings{kryscinski-etal-2020-evaluating,
    title = "Evaluating the Factual Consistency of Abstractive Text Summarization",
    author = "Kryscinski, Wojciech  and
      McCann, Bryan  and
      Xiong, Caiming  and
      Socher, Richard",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.750",
    doi = "10.18653/v1/2020.emnlp-main.750",
    pages = "9332--9346",
}

@inproceedings{lee-etal-2022-masked,
    title = "Masked Summarization to Generate Factually Inconsistent Summaries for Improved Factual Consistency Checking",
    author = "Lee, Hwanhee  and
      Yoo, Kang Min  and
      Park, Joonsuk  and
      Lee, Hwaran  and
      Jung, Kyomin",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-naacl.76",
    doi = "10.18653/v1/2022.findings-naacl.76",
    pages = "1019--1030",
}
```