from itertools import batched
import json
from math import isnan
from struct import pack
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import tkinter as tk
from tkinter import Label, filedialog
import traceback
from typing import List, Dict
from itertools import combinations


def identify_used_modules(code:str, modules:List[str]) -> Dict[str, int]:
    """
    Identify which modules were used how often in the given code snippet.
    
    :param code: Code to analyse.
    :param modules: List of modules to look for.
    :returns: Dictionary mapping module names to how often they were used. 
    """
    used_modules = {}
    lines = code.split('\n')
    
    for line in lines:
        for module in modules:
            if line.find(f' {module}(') != -1 or line.find(f'.{module}(') != -1:
                if module in used_modules.keys():
                    used_modules[module] += 1
                else: 
                    used_modules[module] = 1 
    return used_modules


def calculate_statistics(path:str) -> None:
    """
    Loads the explanans json file and calculates some statistics for it.
    
    :param path: Name of the dictionary containing the files to use.
    """
    try:
        with open(path + "/explanations.json", 'r') as f:
            data = json.load(f)
        try:
            data.pop('Query')
        except:
            print("No query attached!")
            
        print(module for module in data)

        # Confidences
        mean_confidence_per_module = {}
        for module, explanations in data.items():
            mean_confidence_per_module[module] = np.mean([explanation['Confidence'] for explanation in explanations])# if explanation['Confidence'] != 0 and explanation['Alternative Code'][0] != ''])
        print("Confidences: ")
        print(mean_confidence_per_module)
        
        # Ties
        mean_ties_per_module = {}
        for module, explanations in data.items():
            mean_ties = {}
            for other_module in explanations[0]['Ties']:
                mean_ties[other_module] = np.mean([explanation['Ties'][other_module] for explanation in explanations if explanation['Confidence'] != 0])# if explanation['Ties'][other_module] != 0 and explanation['Alternative Code'][0] != ''])
            mean_ties_per_module[module] = mean_ties
        print("Ties: ")
        print(mean_ties_per_module)
        
        # Plot confidence as heatmap
        df = pd.DataFrame([mean_confidence_per_module.values()])
        df.fillna(0, inplace=True)
        df.sort_index(inplace=True)
        df.sort_index(axis=1, inplace=True)
        
        plt.figure(figsize=(10, 6))
        sns.set(font_scale=1.4)
        sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f", cbar=False, square=True,
                        yticklabels=False, xticklabels=mean_confidence_per_module.keys())
        plt.tight_layout()
        plt.yticks(rotation=0)
        plt.title("Confidence")
        plt.savefig("MeanConfidenceHeatmap.pdf")
        plt.show()

        # Plot ties as heatmap
        df = pd.DataFrame.from_dict(mean_ties_per_module, orient='index')
        df.fillna(0, inplace=True)
        df.sort_index(inplace=True)
        df.sort_index(axis=1, inplace=True)
        
        plt.figure(figsize=(10, 6))
        sns.set(font_scale=0.8)
        sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f", square=True)
        plt.tight_layout()
        plt.savefig("MeanTiesHeatmap.pdf")
        plt.show()
        
        """
        for module, ties in mean_ties_per_module.items():
            df = pd.DataFrame([ties.values()])
            df.fillna(0, inplace=True)
            df.sort_index(axis=1, inplace=True)
            
            plt.figure(figsize=(10, 6))
            sns.set(font_scale=1.4)
            sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f", cbar=False, square=True,
                        yticklabels=False, xticklabels=ties.keys())
            plt.tight_layout()
            plt.yticks(rotation=0)
            plt.title(f"Ties from '{module}'")
            plt.savefig(f"MeanTiesHeatmap{module}.pdf")
        plt.show()
        """

        # Plot ties as heatmap (Valdiation bool_to_yesno)
        adapted_ties = {key: mean_ties_per_module[key] for key in ["bool_to_yesno", "verify_property", "exists"]}
        df = pd.DataFrame.from_dict(adapted_ties, orient='index')
        df.fillna(0, inplace=True)
        df.sort_index(inplace=True)
        df.sort_index(axis=1, inplace=True)
        
        plt.figure(figsize=(10, 6))
        sns.set(font_scale=0.9)
        sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f", square=True, cbar_kws={"shrink": 0.35})
        plt.title("Ties from module A to module B", pad=10)
        plt.ylabel("Module A", labelpad=10)
        plt.xlabel("Module B", labelpad=10)
        plt.tight_layout()
        plt.savefig("ValidationMeanTiesHeatmap.pdf")
        plt.show()
    except:
        print("Some error occured.")
        traceback.print_exc()


def display_for_query(path:str, i:int) -> None:
    """
    Displays plots containing explanantions about i-th query.
    
    :param path: Name of the dictionary containing the files to use.
    :param i: Index of the query to display results for.
    """
    try:
        with open(path + "/explanations.json", 'r') as f:
            data = json.load(f)
        try:
            data.pop('Query')
        except:
            print("No query attached!")
            
        print(module for module in data)

        # Confidences
        confidence_per_module = {}
        for module, explanations in data.items():
            confidence_per_module[module] = explanations[i]['Confidence']
        print("Confidences: ")
        print(confidence_per_module)
        
        # Ties
        ties_per_module = {}
        for module, explanations in data.items():
            ties = {}
            for other_module in explanations[0]['Ties']:
                ties[other_module] = explanations[i]['Ties'][other_module]
            ties_per_module[module] = ties
        print("Ties: ")
        print(ties_per_module)
        
        # Plot confidence as heatmap
        df = pd.DataFrame([confidence_per_module.values()])
        df.fillna(0, inplace=True)
        df.sort_index(inplace=True)
        df.sort_index(axis=1, inplace=True)
        
        plt.figure(figsize=(10, 6))
        sns.set(font_scale=1.4)
        sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f", cbar=False, square=True,
                        yticklabels=False, xticklabels=confidence_per_module.keys())
        plt.tight_layout()
        plt.yticks(rotation=0)
        plt.title("Confidence")
        plt.savefig("SingleConfidenceHeatmap.pdf")
        plt.show()

        # Plot ties as heatmap
        df = pd.DataFrame.from_dict(ties_per_module, orient='index')
        df.fillna(0, inplace=True)
        df.sort_index(inplace=True)
        df.sort_index(axis=1, inplace=True)
        
        plt.figure(figsize=(10, 6))
        sns.set(font_scale=0.8)
        sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f", square=True)
        plt.title("Mean Ties from A to B")
        plt.xlabel("Module B")
        plt.ylabel("Module A")
        plt.tight_layout()
        plt.savefig("SingleTiesHeatmap.pdf")
        plt.show()
        
        for module, ties in ties_per_module.items():
            df = pd.DataFrame([ties.values()])
            df.fillna(0, inplace=True)
            df.sort_index(axis=1, inplace=True)
            
            plt.figure(figsize=(10, 6))
            sns.set(font_scale=1.4)
            sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f", cbar=False, square=True,
                        yticklabels=False, xticklabels=ties.keys())
            plt.tight_layout()
            plt.yticks(rotation=0)
            plt.title(f"Ties from '{module}'")
            plt.savefig(f"SingleTiesHeatmap{module}.pdf")
            plt.show()
    except:
        print("Some error occured.")
        traceback.print_exc()


def test_consistency(path:str, explanations_per_sample:int = 10) -> None:
    """
    Calculates consistency of explanations.
    
    :param path: Name of the dictionary containing the files to use.
    :param explanations_per_samples: How many explanations are contained per sample in the file.
    """
    try:
        with open(path + "/explanations.json", 'r') as f:
            data_explanations_dict = json.load(f)
        try:
            data_explanations_dict.pop('Query')
        except:
            print("No query attached!")
            
        number_of_samples = int(len(data_explanations_dict['find']) / explanations_per_sample)
        max_diffs_confidence_unfiltered = []
        max_diffs_confidence = []
        mean_diffs_confidence = []
        max_diffs_ties_unfiltered = []
        max_diffs_ties = []
        mean_diffs_ties = []
        
        # Gather variation in confidences
        for i in range(number_of_samples):
            for module, explanations in data_explanations_dict.items():
                confidences = [explanation['Confidence'] for explanation in explanations[i * explanations_per_sample : (i + 1) * explanations_per_sample]]
                
                diff = np.max(confidences) - np.min(confidences)
                max_diffs_confidence_unfiltered.append(diff)

                if all(confidences) == 0:
                    continue
                
                max_diffs_confidence.append(diff)
                for confidence in confidences:
                    mean_diff = np.abs(np.mean(confidences) - confidence)
                    mean_diffs_confidence.append(mean_diff)
                
        # Gather variation in ties
        for i in range(number_of_samples):
            for module, explanations in data_explanations_dict.items():
                for other_module in data_explanations_dict:
                    ties = [explanation['Ties'][other_module] for explanation in explanations[i * explanations_per_sample : (i + 1) * explanations_per_sample]]
                    
                    diff = np.max(ties) - np.min(ties)
                    max_diffs_ties_unfiltered.append(diff)

                    if all(ties) == 0:
                        continue
                    
                    max_diffs_ties.append(diff)
                    for tie in ties:
                        mean_diff = np.abs(np.mean(ties) - tie)
                        mean_diffs_ties.append(mean_diff)
                
        # Analyse results
        print("\n Unfiltered Confidence Results")
        print("Total Number of Confidence Samples")
        print(len(max_diffs_confidence_unfiltered))
        print("Number of Confidence Samples where the Difference is 0")
        print(len(np.where(np.array(max_diffs_confidence_unfiltered)==0)[0]))
        print("Number of Confidence Samples where the Difference is 0.05 or lower")
        print(len(np.where(np.array(max_diffs_confidence_unfiltered)<=0.05)[0]))
        print("Number of Confidence Samples where the Difference is 0.1 or lower")
        print(len(np.where(np.array(max_diffs_confidence_unfiltered)<=0.1)[0]))
        print("Number of Confidence Samples where the Difference is 0.15 or lower")
        print(len(np.where(np.array(max_diffs_confidence_unfiltered)<=0.15)[0]))
        print("Number of Confidence Samples where the Difference is 0.2 or lower")
        print(len(np.where(np.array(max_diffs_confidence_unfiltered)<=0.2)[0]))
        
        plt.figure(figsize=(7.5,6))
        sns.set_style('whitegrid')            
        sns.set_context("paper", 3, rc={"lines.linewidth": 3})
        sns.kdeplot(np.array(max_diffs_confidence_unfiltered))
        plt.xlim(0, 0.5)
        plt.axvline(x=0.1, color='r', ls='--')    
        plt.axvline(x=0.2, color='r', ls=':')
        plt.xlabel("Max Difference")
        plt.ylabel("Density")
        ax = plt.gca()
        yticks = ax.yaxis.get_major_ticks() 
        yticks[0].label1.set_visible(False)
        plt.savefig("MaxDiffConfidenceUnfiltered.pdf",bbox_inches='tight')
        plt.show()
        
        print("\n Filtered Confidence Results")
        print("Total Number of Non-Zero Confidence Samples")
        print(len(max_diffs_confidence))
        print("Number of Confidence Samples where the Difference is 0")
        print(len(np.where(np.array(max_diffs_confidence)==0)[0]))
        print("Number of Confidence Samples where the Difference is 0.05 or lower")
        print(len(np.where(np.array(max_diffs_confidence)<=0.05)[0]))
        print("Number of Confidence Samples where the Difference is 0.1 or lower")
        print(len(np.where(np.array(max_diffs_confidence)<=0.1)[0]))
        print("Number of Confidence Samples where the Difference is 0.15 or lower")
        print(len(np.where(np.array(max_diffs_confidence)<=0.15)[0]))
        print("Number of Confidence Samples where the Difference is 0.2 or lower")
        print(len(np.where(np.array(max_diffs_confidence)<=0.2)[0]))
        
        plt.figure(figsize=(7.5,6))
        sns.set_style('whitegrid')            
        sns.set_context("paper", 3, rc={"lines.linewidth": 3})
        sns.kdeplot(np.array(max_diffs_confidence))
        plt.xlim(0, 0.5)
        plt.axvline(x=0.1, color='r', ls='--')    
        plt.axvline(x=0.2, color='r', ls=':')
        plt.xlabel("Max Difference")
        plt.ylabel("Density")
        ax = plt.gca()
        yticks = ax.yaxis.get_major_ticks() 
        yticks[0].label1.set_visible(False)
        plt.savefig("MaxDiffConfidence.pdf",bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(7.5,6))
        sns.set_style('whitegrid')            
        sns.set_context("paper", 3, rc={"lines.linewidth": 3})
        sns.kdeplot(np.array(mean_diffs_confidence))
        plt.xlim(0, 0.5)
        plt.axvline(x=0.1, color='r', ls='--')    
        plt.axvline(x=0.2, color='r', ls=':')
        plt.xlabel("Mean Difference")
        plt.ylabel("Density")
        ax = plt.gca()
        yticks = ax.yaxis.get_major_ticks() 
        yticks[0].label1.set_visible(False)
        plt.savefig("MeanDiffConfidence.pdf",bbox_inches='tight')
        plt.show()
        
        print("\n Unfiltered Ties Results")
        print("Total Number of Ties Samples")
        print(len(max_diffs_ties_unfiltered))
        print("Number of Ties Samples where the Difference is 0")
        print(len(np.where(np.array(max_diffs_ties_unfiltered)==0)[0]))
        print("Number of Ties Samples where the Difference is 0.05 or lower")
        print(len(np.where(np.array(max_diffs_ties_unfiltered)<=0.05)[0]))
        print("Number of Ties Samples where the Difference is 0.1 or lower")
        print(len(np.where(np.array(max_diffs_ties_unfiltered)<=0.1)[0]))
        print("Number of Ties Samples where the Difference is 0.15 or lower")
        print(len(np.where(np.array(max_diffs_ties_unfiltered)<=0.15)[0]))
        print("Number of Ties Samples where the Difference is 0.2 or lower")
        print(len(np.where(np.array(max_diffs_ties_unfiltered)<=0.2)[0]))

        plt.figure(figsize=(7.5,6))
        sns.set_style('whitegrid')            
        sns.set_context("paper", 3, rc={"lines.linewidth": 3})
        sns.kdeplot(np.array(max_diffs_ties_unfiltered))
        plt.xlim(0, 0.5)
        plt.axvline(x=0.1, color='r', ls='--')
        plt.axvline(x=0.2, color='r', ls=':')
        plt.xlabel("Max Difference")
        plt.ylabel("Density")
        ax = plt.gca()
        yticks = ax.yaxis.get_major_ticks() 
        yticks[0].label1.set_visible(False)
        plt.savefig("MaxDiffTiesUnfiltered.pdf",bbox_inches='tight')
        plt.show()
        
        print("\n Filtered Ties Results")
        print("Total Number of Non-Zero Ties Samples")
        print(len(max_diffs_ties))
        print("Number of Ties Samples where the Difference is 0")
        print(len(np.where(np.array(max_diffs_ties)==0)[0]))
        print("Number of Ties Samples where the Difference is 0.05 or lower")
        print(len(np.where(np.array(max_diffs_ties)<=0.05)[0]))
        print("Number of Ties Samples where the Difference is 0.1 or lower")
        print(len(np.where(np.array(max_diffs_ties)<=0.1)[0]))
        print("Number of Ties Samples where the Difference is 0.15 or lower")
        print(len(np.where(np.array(max_diffs_ties)<=0.15)[0]))
        print("Number of Ties Samples where the Difference is 0.2 or lower")
        print(len(np.where(np.array(max_diffs_ties)<=0.2)[0]))

        plt.figure(figsize=(7.5,6))
        sns.set_style('whitegrid')            
        sns.set_context("paper", 3, rc={"lines.linewidth": 3})
        sns.kdeplot(np.array(max_diffs_ties))
        plt.xlim(0, 0.5)
        plt.axvline(x=0.1, color='r', ls='--')
        plt.axvline(x=0.2, color='r', ls=':')
        plt.xlabel("Max Difference")
        plt.ylabel("Density")
        ax = plt.gca()
        yticks = ax.yaxis.get_major_ticks() 
        yticks[0].label1.set_visible(False)
        plt.savefig("MaxDiffTies.pdf",bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(7.5,6))
        sns.set_style('whitegrid')            
        sns.set_context("paper", 3, rc={"lines.linewidth": 3})
        sns.kdeplot(np.array(mean_diffs_ties))
        plt.xlim(0, 0.5)
        plt.axvline(x=0.1, color='r', ls='--')    
        plt.axvline(x=0.2, color='r', ls=':')
        plt.xlabel("Mean Difference")
        plt.ylabel("Density")
        ax = plt.gca()
        yticks = ax.yaxis.get_major_ticks() 
        yticks[0].label1.set_visible(False)
        plt.savefig("MeanDiffConfidence.pdf",bbox_inches='tight')
        plt.show()

        print("\n Average Jaccard Distance")
        all_modules = list(data_explanations_dict)
        jaccard_dists = []
        for module, explanations in data_explanations_dict.items():
            for explanation in explanations:
                if explanation['Alternative Code'][0] == '':
                    continue

                used_modules = [set(identify_used_modules(alt_code, all_modules)) for alt_code in explanation['Alternative Code']]
                for a, b in combinations(used_modules, 2):
                    try:
                        jaccard = 1 - len(a & b) / len(a | b)
                        jaccard_dists.append(jaccard)
                    except:
                        continue
        print(np.mean(jaccard_dists))
            
    except:
        print("Some error occured.")
        traceback.print_exc()


def accuracy(path:str) -> None:
    """
    Calculates accuracy of given results file.
    
    :param path: Name of the dictionary containing the files to use.
    """
    try:
        data_results_df = pd.read_csv(path + "/results.csv")
            
        import dataset_helpers
        results = [data_point["result"] for idx, data_point in data_results_df.iterrows()]
        answers = [data_point["answer"] for idx, data_point in data_results_df.iterrows()]
        accuracy = dataset_helpers.accuracy(results, answers)
        print("Original")
        print(accuracy)    
        
        default_answers = len([0 for idx, data_point in data_results_df.iterrows() if 'image_patch.simple_query(query)' in data_point["code"] or 
                                                                                        ".forward('glip'" in data_point["code"]])
        print("Non-executable Code")
        print(default_answers/len(answers))

        try:
            alt_results = [data_point["alt_result"] for idx, data_point in data_results_df.iterrows()]
            accuracy = dataset_helpers.accuracy(alt_results, answers)
            print("Following recommendation")
            print(accuracy)
            
            default_answers = len([0 for idx, data_point in data_results_df.iterrows() if 'image_patch.simple_query(query)' in data_point["alt_code"] or 
                                                                                        ".forward('glip'" in data_point["alt_code"]])
            print("Non-executable Code")
            print(default_answers/len(answers))
        except:
            print("No alt results")
        
    except:
        print("Some error occured.")
        traceback.print_exc()
        

def test_correlation(path:str, answers_per_sample:int = 10) -> None:
    """
    Calculates correlation betweem accuracy improvement and confidence.
    Assumes ratio of answers to explanations to be answers_per_sample to 1.
    
    :param path: Name of the dictionary containing the files to use.
    :answers_per_sample: How many answers to expect per query.
    """
    try:
        # Load Data.
        data_results_df = pd.read_csv(path + "/results.csv")
        
        with open(path + "/explanations.json", 'r') as f:
            data_explanations_dict = json.load(f)
            
        all_queries = data_results_df.get('query')
        def extract_ranges(lst):
            result = []
            step = 200
            length = 20
            for start in range(0, len(lst), step):
                end = start + length
                result.extend(lst[start:end])
            return result
        all_queries = extract_ranges(all_queries)
        all_modules = list(data_explanations_dict)
        all_modules.pop(all_modules.index('Query'))

        # Sort result files by query.
        import dataset_helpers
        correctness_values_per_task = {query:[] for query in all_queries}
        used_modules_per_task = {query:[] for query in all_queries}
        for idx, data_point in data_results_df.iterrows():
            correctness_values_per_task[data_point["query"]].append(dataset_helpers.accuracy([data_point["result"]], [data_point["answer"]]))
            used_modules_per_task[data_point["query"]].append(identify_used_modules(data_point["code"], all_modules))

        # Sort explanations by query.
        explanations_per_task = {query:[] for query in all_queries}
        for i in range(len(data_explanations_dict['find'])):
            query = all_queries[int(i / 10)]  #data_explanations_dict['Query'][i]
            explanations_per_task[query].append({module:data_explanations_dict[module][i] for module in all_modules})
            
        # Get confidence differences in answers to query.
        all_confidence_diffs = []
        for query, explanations in explanations_per_task.items():
            mean_confidences = [] 
            for i in range(len(correctness_values_per_task[query])):
                mean_confidence_of_used = float(np.mean([explanation['Confidence'] for module, explanation in explanations[-1].items() if module in used_modules_per_task[query][i]]))
                mean_confidence_of_used = np.nan_to_num(mean_confidence_of_used, nan=0.0)
                mean_confidences.append(mean_confidence_of_used)
            if min(mean_confidences) > 0:
                confidence_diffs = [confidence - min(mean_confidences) for confidence in mean_confidences]
            else:
                confidence_diffs = [np.nan for confidence in mean_confidences]
            all_confidence_diffs.extend(confidence_diffs)

        # Calculate correlation.
        from scipy.stats import pointbiserialr
        all_correctness_values = []
        for task, correctness_values in correctness_values_per_task.items():
            all_correctness_values.extend(correctness_values)
        print(len(all_confidence_diffs))
        print(len(all_correctness_values))
        #mask = ~np.isnan(all_confidence_diffs) & ~np.isnan(all_correctness_values)
        mask = np.where((np.array(all_confidence_diffs) <= 0.47) | (np.array(all_correctness_values) == 1))[0]
        all_confidence_diffs = np.array(all_confidence_diffs)[mask]
        all_correctness_values = np.array(all_correctness_values)[mask]
        mask = ~np.isnan(all_confidence_diffs)
        all_confidence_diffs = all_confidence_diffs[mask]
        all_correctness_values = all_correctness_values[mask]#TODO: Clean this up
        print(len(all_confidence_diffs))
        print(pointbiserialr(all_confidence_diffs, all_correctness_values))
        mask = np.where(all_confidence_diffs != 0)[0]
        all_confidence_diffs = all_confidence_diffs[mask]
        all_correctness_values = all_correctness_values[mask]
        print(len(all_confidence_diffs))
        print(pointbiserialr(all_confidence_diffs, all_correctness_values))
        
        df = pd.DataFrame({
            'Mean Confidence': all_confidence_diffs,
            'Correctness': ["Correct" if value else "Incorrect" for value in all_correctness_values]
        })
        plt.figure(figsize=(5,4))
        sns.set_style('whitegrid')      
        sns.boxplot(y='Mean Confidence', x='Correctness', data=df, width=0.4, 
                    boxprops=dict(facecolor="turquoise"), showfliers = False)
        sns.set_style('whitegrid')      
        plt.ylabel('Improvement of Mean Confidence')
        plt.xlabel('')
        plt.savefig("MeanConfidenceCorrelation.pdf",bbox_inches='tight')
        plt.show()
        
    except:
        print("Some error occured.")
        traceback.print_exc()


if __name__ == "__main__":
    path = filedialog.askdirectory(title="Choose working directory containing 'explanations.json' and 'results.csv'.")
    root = tk.Tk()
    root.geometry("300x300+50+50")
    btns = []
    samplesize_widget = tk.Entry(root)
    samplesize_widget.insert(0, "10")
    def resetPath():
        global path
        path = filedialog.askdirectory(title="Choose working directory containing 'explanations.json' and 'results.csv'.")
    btns.append(tk.Button(root, text="Reset Path", command=lambda:resetPath()))
    btns.append(tk.Button(root, text="Stats", command=lambda:calculate_statistics(path)))
    btns.append(tk.Button(root, text="StatsSingular", command=lambda:display_for_query(path, int(samplesize_widget.get()))))
    btns.append(tk.Button(root, text="Correlation", command=lambda:test_correlation(path, int(samplesize_widget.get()))))
    btns.append(tk.Button(root, text="Consistency", command=lambda:test_consistency(path, int(samplesize_widget.get()))))
    btns.append(tk.Button(root, text="Accuracy", command=lambda:accuracy(path)))
    for btn in btns:
        btn.pack(side=tk.TOP, pady=5)
    tk.Label(root, text="sample size (if needed):").pack(side=tk.TOP, pady=0)
    samplesize_widget.pack(side=tk.TOP, pady=5)
    root.mainloop()
