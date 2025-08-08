from itertools import batched
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


def calculate_statistics(filename: str = "explanans.json") -> None:
    """
    Loads the explanans json file and calculates some statistics for it.
    
    :param filename: Name of the file to load.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
        
    # ###############
    #  Confidences
    # ###############
    mean_confidence_per_module = {}
    for module, explanations in data.items():
        mean_confidence_per_module[module] = np.mean([explanation['Confidence'] for explanation in explanations if explanation['Confidence'] != 0])
    print("Confidences: ")
    print(mean_confidence_per_module)
    
    # ###############
    #  Ties
    # ###############
    mean_ties_per_module = {}
    for module, explanations in data.items():
        mean_ties = {}
        for other_module in explanations[0]['Ties']:
            mean_ties[other_module] = np.mean([explanation['Ties'][other_module] for explanation in explanations if explanation['Ties'][other_module] != 0])
        mean_ties_per_module[module] = mean_ties
    print("Ties: ")
    print(mean_ties_per_module)
    
    # Plot ties as heatmap
    df = pd.DataFrame.from_dict(mean_ties_per_module, orient='index')
    df.fillna(0, inplace=True)
    df.sort_index(inplace=True)
    df.sort_index(axis=1, inplace=True)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Mean Ties from A to B")
    plt.xlabel("Module B")
    plt.ylabel("Module A")
    plt.tight_layout()
    plt.show()
    plt.savefig("TiesHeatmap.pdf")


def calculate_correlation(filename_results, filename_explanations, batches = 100):
    """
    Calculates correlation betweem accuracy and confidence.
    
    :param filename_results: Name of the results file to load.
    :param filename_explanations: Name of the explanation file to load.
    """
    data_results_df = pd.read_csv(filename_results)
    
    with open(filename_explanations, 'r') as f:
        data_explanations_dict = json.load(f)
        
    batched_results_df = np.split(data_results_df, batches)
    import dataset_helpers
    batched_accuracies = [dataset_helpers.accuracy(batch["result"], batch["answer"]) for batch in batched_results_df]
    
    mean_confidence_per_result = []
    for i in range(len(data_explanations_dict['find'])):
        mean_confidence = np.mean([explanations[i]['Confidence'] for module, explanations in data_explanations_dict.items() if module != ''])
        mean_confidence_per_result.append(mean_confidence)
    batched_confidences = np.split(mean_confidence_per_result)
    batched_confidence_mean = [np.mean(batch) for batch in batched_confidences]

    plt.figure(figsize=(10, 6))
    plt.plot(batched_confidence_mean, batched_accuracies)
    plt.title("Correlation between Confidence and Accuracy")
    plt.xlabel("Mean Confidence")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()
    plt.savefig("Correlation.pdf")

    from scipy.stats import pearsonr
    print(pearsonr(batched_confidence_mean, batched_accuracies))

if __name__ == "__main__":
    calculate_statistics("explanans.json")
    #calculate_correlation("current-expl-0-24-accuracy.csv", "explanans.json")
