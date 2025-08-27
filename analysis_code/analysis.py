from itertools import batched
import json
from math import isnan
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog


def calculate_statistics(path:str) -> None:
    """
    Loads the explanans json file and calculates some statistics for it.
    
    :param path: Name of the dictionary containing the files to use.
    """
    with open(path + "/explanations.json", 'r') as f:
        data = json.load(f)
        
    # Confidences
    mean_confidence_per_module = {}
    for module, explanations in data.items():
        mean_confidence_per_module[module] = np.mean([explanation['Confidence'] for explanation in explanations if explanation['Confidence'] != 0 and explanation['Alternative Code'][0] != ''])
    print("Confidences: ")
    print(mean_confidence_per_module)
    
    # Ties
    mean_ties_per_module = {}
    for module, explanations in data.items():
        mean_ties = {}
        for other_module in explanations[0]['Ties']:
            mean_ties[other_module] = np.mean([explanation['Ties'][other_module] for explanation in explanations if explanation['Ties'][other_module] != 0 and explanation['Alternative Code'][0] != ''])
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
    plt.savefig("TiesHeatmap.pdf")
    plt.show()


def calculate_correlation(path:str) -> None:
    """
    Calculates correlation betweem accuracy and confidence.
    
    :param path: Name of the dictionary containing the files to use.
    """
    data_results_df = pd.read_csv(path + "/results.csv")
    
    with open(path + "/explanations.json", 'r') as f:
        data_explanations_dict = json.load(f)
        
    import dataset_helpers
    correctness_values = []
    invalid = []
    for idx, data_point in data_results_df.iterrows():
        #if "simple_query(query)" not in data_point["code"] and "ImagePatch" not in data_point["answer"] and "yes" not in data_point["answer"] and "no" not in data_point["answer"]:
        #if "simple_query(query)" not in data_point["code"] and "ImagePatch" not in data_point["answer"] and ("yes" in data_point["answer"] or "no" in data_point["answer"]):
        if "simple_query(query)" not in data_point["code"] and "ImagePatch" not in data_point["answer"]:
            correctness_values.append(dataset_helpers.accuracy([data_point["result"]], [data_point["answer"]]))
        else:
            correctness_values.append(-1)
            invalid.append(idx)

    print(len(data_explanations_dict['find']))

    mean_confidence_alt_per_result = []
    min_confidence_alt_per_result = []
    max_confidence_alt_per_result = []
    mean_confidence_per_result = []
    min_confidence_per_result = []
    max_confidence_per_result = []
    for i in range(len(data_explanations_dict['find'])):
        if i in invalid:
            continue
        
        if len([explanations[i]['Alternative Code'] for module, explanations in data_explanations_dict.items() if explanations[i]['Alternative Code'][0] != '']) > 1:
            
            confidences = [explanations[i]['Confidence'] for module, explanations in data_explanations_dict.items() if explanations[i]['Alternative Code'][0] != '']
            mean_confidence = np.mean(confidences) if len(confidences) != 0 else 0
            min_confidence = np.min(confidences) if len(confidences) != 0 else 0
            max_confidence = np.max(confidences) if len(confidences) != 0 else 0
           
            mean_confidence_per_result.append(mean_confidence)
            min_confidence_per_result.append(min_confidence)
            max_confidence_per_result.append(max_confidence)
            
            
            confidences_alt = [explanations[i]['Confidence'] for module, explanations in data_explanations_dict.items() if explanations[i]['Alternative Code'][0] == '']
            mean_confidence_alt = np.mean(confidences_alt) if len(confidences_alt) != 0 else 0
            min_confidence_alt = np.min(confidences_alt) if len(confidences_alt) != 0 else 0
            max_confidence_alt = np.max(confidences_alt) if len(confidences_alt) != 0 else 0
           
            mean_confidence_alt_per_result.append(mean_confidence_alt)
            min_confidence_alt_per_result.append(min_confidence_alt)
            max_confidence_alt_per_result.append(max_confidence_alt)
        else:
            invalid.append(i)
            
    correctness_values = [correctness_values[i] for i in range(len(correctness_values)) if i not in invalid]

    print(len(mean_confidence_per_result))

    from scipy.stats import pointbiserialr
    print(pointbiserialr(mean_confidence_per_result, correctness_values))
    print(pointbiserialr(min_confidence_per_result, correctness_values))
    print(pointbiserialr(max_confidence_per_result, correctness_values))
    
    df = pd.DataFrame({
        'Mean Confidence': mean_confidence_per_result,
        'Correctness': ["Correct" if value else "Incorrect" for value in correctness_values]
    })
    sns.boxplot(y='Mean Confidence', x='Correctness', data=df)
    plt.ylabel('Mean Confidence of Used Modules')
    plt.xlabel('')
    plt.title('Mean Confidence by Correctness')
    plt.savefig("MeanConfidenceCorrelation.pdf")
    plt.show()
    
    df = pd.DataFrame({
        'Min Confidence': min_confidence_per_result,
        'Correctness': ["Correct" if value else "Incorrect" for value in correctness_values]
    })
    sns.boxplot(y='Min Confidence', x='Correctness', data=df)
    plt.ylabel('Minimum Confidence of Used Modules')
    plt.xlabel('')
    plt.title('Min Confidence by Correctness')
    plt.savefig("MinConfidenceCorrelation.pdf")
    plt.show()
    

def test_consistency(path:str, explanations_per_sample:int = 10) -> None:
    """
    Calculates consistency of explanations.
    
    :param path: Name of the dictionary containing the files to use.
    :param explanations_per_samples: How many explanations are contained per sample in the file.
    """
    from scipy.stats import variation  # coefficient_of_variation
    with open(path + "/explanations.json", 'r') as f:
        data_explanations_dict = json.load(f)
        
    number_of_samples = int(len(data_explanations_dict['find']) / explanations_per_sample)
    variations_confidence = []
    diffs_confidence = []
    variations_ties = []
    diffs_ties = []
    
    for i in range(number_of_samples):
        for module, explanations in data_explanations_dict.items():
            confidences = [explanation['Confidence'] for explanation in explanations[i * explanations_per_sample : (i + 1) * explanations_per_sample]]
            
            if all(confidences) == 0:
                continue

            var = variation(confidences, ddof=1) * 100
            if np.isnan(var):
                var = 0
            variations_confidence.append(var)
            
            diff = np.max(confidences) - np.min(confidences)
            diffs_confidence.append(diff)
            
    for i in range(number_of_samples):
        for module, explanations in data_explanations_dict.items():
            for other_module in data_explanations_dict:
                ties = [explanation['Ties'][other_module] for explanation in explanations[i * explanations_per_sample : (i + 1) * explanations_per_sample]]
                
                if all(ties) == 0:
                    continue

                var = variation(ties, ddof=1) * 100
                if np.isnan(var):
                    var = 0
                variations_ties.append(var)
                
                diff = np.max(ties) - np.min(ties)
                diffs_ties.append(diff)
            
    print("Total Number of Non-Zero Confidence Samples")
    print(len(diffs_confidence))
    print("Number of Confidence Samples where the Difference is 0")
    print(len(np.where(np.array(diffs_confidence)==0)[0]))
    print("Number of Confidence Samples where the Difference is 0.1 or lower")
    print(len(np.where(np.array(diffs_confidence)<=0.1)[0]))
    
    sns.set_style('whitegrid')            
    sns.kdeplot(np.array(diffs_confidence))
    plt.xlim(0, 0.5)
    plt.axvline(x=0.1, color='r', ls=':')
    plt.xlabel("Max Difference")
    plt.title("KDE Max Difference of Confidence Measurements between Different Explanations")
    plt.savefig("MaxDiffConfidence.pdf")
    plt.show()
    
    print("Total Number of Non-Zero Ties Samples")
    print(len(diffs_ties))
    print("Number of Ties Samples where the Difference is 0")
    print(len(np.where(np.array(diffs_ties)==0)[0]))
    print("Number of Ties Samples where the Difference is 0.1 or lower")
    print(len(np.where(np.array(diffs_ties)<=0.1)[0]))

    sns.kdeplot(np.array(diffs_ties))
    plt.xlim(0, 0.5)
    plt.axvline(x=0.1, color='r', ls=':')
    plt.xlabel("Max Difference")
    plt.title("KDE Max Difference of Ties Measurements between Different Explanations")
    plt.savefig("MaxDiffTies.pdf")
    plt.show()


def accuracy(path:str):
    """
    Calculates accuracy of given results file.
    
    :param path: Name of the dictionary containing the files to use.
    """
    data_results_df = pd.read_csv(path + "/results.csv")
        
    import dataset_helpers
    results = [data_point["result"] for idx, data_point in data_results_df.iterrows()]
    answers = [data_point["answer"] for idx, data_point in data_results_df.iterrows()]
    accuracy = dataset_helpers.accuracy(results, answers)
    print(accuracy)



if __name__ == "__main__":
    path = filedialog.askdirectory(title="Choose working directory containing 'explanations.json' and 'results.csv'.")
    root = tk.Tk()
    root.geometry("300x300+50+50")
    btns = []
    def resetPath():
        global path
        path = filedialog.askdirectory(title="Choose working directory containing 'explanations.json' and 'results.csv'.")
    btns.append(tk.Button(root, text="Reset Path", command=lambda:resetPath()))
    btns.append(tk.Button(root, text="Stats", command=lambda:calculate_statistics(path)))
    btns.append(tk.Button(root, text="Correlation", command=lambda:calculate_correlation(path)))
    btns.append(tk.Button(root, text="Consistency", command=lambda:test_consistency(path)))
    btns.append(tk.Button(root, text="Accuracy", command=lambda:accuracy(path)))
    for btn in btns:
        btn.pack(side=tk.TOP, pady=5)
    root.mainloop()
