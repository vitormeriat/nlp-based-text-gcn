from os import listdir
from os.path import isfile, join
import pandas as pd
import sys
import numpy
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt

PATH = "./"
DATASETS = ["MR", "Ohsumed", "R8", "R52"]
MODES = ["default", "syntactic", "semantic", "graph"]


mode = "default"

dataset_config = {
    "MR": {"marker": 'o', "linestyle": '-'},
    "Ohsumed": {"marker": '^', "linestyle": '--'},
    "R8": {"marker": 's', "linestyle": '-.'},
    "R52": {"marker": 'D', "linestyle": ':'}
}

basic_stats = {}

for dataset in DATASETS:
    mypath = f'statistics/{mode}/{dataset}.csv'
    basic_stats = pd.read_csv(mypath, delimiter=';')

    plt.plot(basic_stats["name"], basic_stats["mean"],
             marker=dataset_config[dataset]["marker"],
             linestyle=dataset_config[dataset]["linestyle"], label=dataset)

    for x, y in zip(basic_stats["name"], basic_stats["mean"]):

        label = "{:.4f}".format(y)

        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center',  # horizontal alignment can be left, right or center
                     rotation=15,
                     fontsize=18)

    # plt.title("Average accurracy of each experiment",fontsize=36)
    plt.xticks(rotation=30, fontsize=20)
    plt.yticks(numpy.arange(0.65, 1.05, step=0.05), fontsize=20)
    plt.legend(loc='lower center', fontsize=20, mode=None,
               ncol=4, bbox_to_anchor=(0.5, -0.22))
    plt.savefig(f'statistics/{mode}/{mode.upper()}.png', dpi=300)
    plt.close()


def calculate_statistics(max_result, dataset):
    ORIGINAL_PAPER = {
        "MR": {"avg": 0.7674, "std": 0.0020, "n": 10},
        "Ohsumed": {"avg": 0.6836, "std": 0.0056, "n": 10},
        "R8": {"avg": 0.9707, "std": 0.0010, "n": 10},
        "R52": {"avg": 0.9356, "std": 0.0018, "n": 10}
    }

    OBS = {
        "y1": {  # Original paper
            "mean": ORIGINAL_PAPER[dataset]["avg"],
            "std": ORIGINAL_PAPER[dataset]["std"],
            "var": ORIGINAL_PAPER[dataset]["std"] ** 2,
            "n": ORIGINAL_PAPER[dataset]["n"]
        },
        "y2": {  # Best result
            "mean": max_result["mean"].values[0],
            "std":  max_result["standard_deviation"].values[0],
            "var": max_result["standard_deviation"].values[0] ** 2,
            "n": 10

        }
    }

    max_std = numpy.max([OBS["y1"]["std"], OBS["y2"]["std"]])
    min_std = numpy.min([OBS["y1"]["std"], OBS["y2"]["std"]])

    IS_VARIANCE_EQUAL = max_std/min_std <= 2

    ttest_ind_from_stats = stats.ttest_ind_from_stats(
        mean2=max_result["mean"].values[0],
        std2=max_result["standard_deviation"].values[0],
        nobs2=10,
        mean1=ORIGINAL_PAPER[dataset]["avg"],
        std1=ORIGINAL_PAPER[dataset]["std"],
        nobs1=10,
        equal_var=IS_VARIANCE_EQUAL
    )

    pvalue = ttest_ind_from_stats.pvalue
    alfa = 0.03

    test_result = "Reject H0" if pvalue <= alfa else "Fail to reject H0"

    return({
        "Dataset": dataset,
        "Experiment": round(max_result["experiment"].values[0], 4),
        "Experiment Name": max_result["name"].values[0],
        "Experiment Mean": round(max_result["mean"].values[0], 4),
        "Experiment Std": round(max_result["standard_deviation"].values[0], 4),
        "Original Mean": round(ORIGINAL_PAPER[dataset]["avg"], 4),
        "Original Std": round(ORIGINAL_PAPER[dataset]["std"], 4),
        "T Test Compared Results": test_result,
        "Alfa": alfa,
        "T-value": round(ttest_ind_from_stats.statistic, 4),
        "P-value": pvalue,
        "Level of Confidence (%)": round(((1-pvalue)*100), 4),
        "Higher Mean": "Original" if max_result["mean"].values[0] < ORIGINAL_PAPER[dataset]["avg"] else "Experiment" if max_result["mean"].values[0] > ORIGINAL_PAPER[dataset]["avg"] else "Equal",
        "Difference (Experiment-Original)": round(max_result["mean"].values[0] - ORIGINAL_PAPER[dataset]["avg"], 4),
        "Equal Variance?": IS_VARIANCE_EQUAL
    })


statistics_results = []
best_results = []

for dataset in DATASETS:
    all_results = basic_stats[dataset]
    for experiment in range(19):
        results = pd.DataFrame(all_results)
        experiment_results = results.loc[(results["experiment"] == experiment)]
        statistics_results.append(
            calculate_statistics(experiment_results, dataset))

    results = pd.DataFrame(all_results)
    max_result = results.loc[(results["mean"] == numpy.max(results["mean"]))]
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


print("Done!!!")
