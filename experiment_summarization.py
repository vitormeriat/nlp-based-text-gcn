from os import listdir
from os.path import isfile, join
import pandas as pd
import sys
import numpy
from scipy import stats
import os


#MODES = ["default", "syntactic", "semantic", "graph"]
DATASETS = ["MR", "Ohsumed", "R8", "R52"]
MODES = ["default", "graph", "syntactic"]


def clean_text(txt):
    return txt.split(':')[1].replace(',', '')


def get_metrics(file_path, info, mode, dataset):
    # epochs = 0
    with open(file_path) as f:
        text = f.readlines()
    for line in text:

        if line[:6] == "Epoch:":
            #line = line.replace('\n', '').replace(', ', ' ').replace(':', ' ').replace('=', ' ')
            epochs = line.split(" ")[1]
        if line[:17] == "Test set results:":
            #line = line.replace('\n', '').replace(', ', ' ').replace(':', ' ').replace('=', ' ')
            tokens = line.split(" ")
            return {
                "cost": tokens[4].replace(',', ''),
                "accuracy": tokens[6].replace(',', ''),
                "epochs": int(epochs),
                "mode": mode,
                "dataset": dataset,
                "experiment": info[1],
                "name": info[4],
                "run": info[6].split(".")[0]
            }


def calculate_basic_statistics(df, mode, dataset):
    # Implment Basic statistics and save as csv

    all_results = []
    exp = []
    for i in range(19):  # This for controls the experiments. Use 25 to get all CUSTOM
        experiment = df.loc[df['experiment'] == i]
        experiment_name = df.loc[df['experiment'] == i]["name"].values[0]
        exp.append(experiment["accuracy"])
        results = {
            'experiment': i,
            'name': experiment_name,
            'min': numpy.min(experiment["accuracy"]),
        }

        results["max"] = numpy.max(experiment["accuracy"])
        results["mean"] = numpy.mean(experiment["accuracy"])
        results["median"] = numpy.median(experiment["accuracy"])
        results["variance"] = numpy.var(experiment["accuracy"])
        results["standard_deviation"] = numpy.std(experiment["accuracy"])
        results["min_epochs"] = numpy.min(experiment["epochs"])
        results["max_epochs"] = numpy.max(experiment["epochs"])
        results["mean_epochs"] = numpy.mean(experiment["epochs"])
        results["median_epochs"] = numpy.median(experiment["epochs"])
        results["var_epochs"] = numpy.var(experiment["epochs"])
        results["std_epochs"] = numpy.std(experiment["epochs"])
        all_results.append(results)

    results = pd.DataFrame(all_results)

    if not os.path.exists(f'./statistics/{mode}/'):
        os.makedirs(f'./statistics/{mode}/')

    results.to_csv(
        # "./Basic_Statistics/"+dataset+".csv",
        f'./statistics/{mode}/{dataset}.csv',
        sep=';',
        index=False
    )
    return results


def calculate_statistics(all_results, dataset):

    print("------ Working with dataset", dataset, "------\n")
    ORIGINAL_PAPER = {
        "MR": {"avg": 0.7674, "std": 0.0020},
        "Ohsumed": {"avg": 0.6836, "std": 0.0056},
        "R8": {"avg": 0.9707, "std": 0.0010},
        "R52": {"avg": 0.9356, "std": 0.0018}
    }

    results = pd.DataFrame(all_results)
    max_result = results.loc[(results["mean"] == numpy.max(results["mean"]))]
    print(results)
    print(max_result)

    ttest_ind_from_stats = stats.ttest_ind_from_stats(
        mean2=max_result["mean"].values[0],
        std2=max_result["standard_deviation"].values[0],
        nobs2=10,
        mean1=ORIGINAL_PAPER[dataset]["avg"],
        std1=ORIGINAL_PAPER[dataset]["std"],
        nobs1=10
    )

    print(ttest_ind_from_stats)


all_results = []
for mode in MODES:
    for dataset in DATASETS:
        for run in range(1):
            #mypath = PATH+dataset+"/RUN_"+str(run)
            # dataset+"/RUN_"+str(run)
            mypath = f'experiments/{mode}/{dataset}/RUN_{run}'
            onlyfiles = [f for f in listdir(mypath) if isfile(
                join(mypath, f)) and f[-4:] == ".txt"]
            # EXPERIMENT_11_model_mr_DO05_run_6.txt
            for f in onlyfiles:
                info = f.split("_")
                # print(info)
                all_results.append(get_metrics(
                    mypath+"/"+f, info, mode, dataset))

df = pd.DataFrame(data=all_results)

df["cost"] = pd.to_numeric(df["cost"])
df["accuracy"] = pd.to_numeric(df["accuracy"])
df["experiment"] = pd.to_numeric(df["experiment"])
df["run"] = pd.to_numeric(df["run"])

df = df.sort_values(by=['experiment', 'run', 'mode', 'dataset'])
df = df.reset_index(drop=True)

statistics_results = []
best_results = []

for mode in MODES:
    for dataset in DATASETS:
        #experiments = df.loc[df['dataset'] == dataset]
        experiments = df.loc[(df['mode'] == mode) & (df['dataset'] == dataset)]
        calculate_basic_statistics(experiments, mode, dataset)

        for experiment in range(19):
            experiment_results = df.loc[(df["experiment"] == experiment)]
            statistics_results.append(
                calculate_statistics(experiment_results, dataset))
        results = pd.DataFrame(statistics_results)
        max_result = results.loc[(
            results["mean"] == numpy.max(results["mean"]))]
        best_results.append(calculate_statistics(max_result, dataset))

    statistics_results_df = pd.DataFrame(statistics_results)
    best_results_df = pd.DataFrame(best_results)

    statistics_results_df.to_csv(
        "./statistics/all_results.csv",
        sep=';',
        index=False
    )

    best_results_df.to_csv(
        "./statistics/best_results.csv",
        sep=';',
        index=False
    )


#statistics_results = []
#best_results = []

# for dataset in DATASETS:
#     all_results = basic_stats[dataset]
#     for experiment in range(19):
#         results = pd.DataFrame(all_results)
#         experiment_results = results.loc[(results["experiment"] == experiment)]
#         statistics_results.append(calculate_statistics(experiment_results, dataset))

#     results = pd.DataFrame(all_results)
#     max_result = results.loc[(results["mean"] == numpy.max(results["mean"]))]
#     best_results.append(calculate_statistics(max_result, dataset))


# statistics_results_df = pd.DataFrame(statistics_results)
# best_results_df = pd.DataFrame(best_results)


# statistics_results_df.to_csv(
#     "./statistics/all_results.csv",
#     sep=';',
#     index=False
# )

# best_results_df.to_csv(
#     "./statistics/best_results.csv",
#     sep=';',
#     index=False
# )
