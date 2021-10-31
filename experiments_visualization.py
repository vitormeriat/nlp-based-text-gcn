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
    #basic_stats[dataset] = df

    #plt.plot(basic_stats[dataset]["name"], basic_stats[dataset]["mean"], marker=dataset_config[dataset]["marker"], linestyle=dataset_config[dataset]["linestyle"], label=dataset)
    plt.plot(basic_stats["name"], basic_stats["mean"],
            marker=dataset_config[dataset]["marker"], linestyle=dataset_config[dataset]["linestyle"], label=dataset)

    for x, y in zip(basic_stats["name"], basic_stats["mean"]):

        label = "{:.4f}".format(y)

        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center',  # horizontal alignment can be left, right or center
                     rotation=15,
                     fontsize=18
                     )

    # plt.title("Average accurracy of each experiment",fontsize=36)
    plt.xticks(rotation=30, fontsize=20)
    plt.yticks(numpy.arange(0.65, 1.05, step=0.05), fontsize=20)
    plt.legend(loc='lower center', fontsize=20, mode=None, ncol=4, bbox_to_anchor=(0.5, -0.22))
    #plt.show()
    plt.savefig(f'statistics/{mode}/{mode.upper()}.png', dpi=300)
    plt.close()


print("Done!!!")
