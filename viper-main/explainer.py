from typing import Dict, List, Any, Tuple
#import main_simple_lib as viperGPT
import numpy as np
import matplotlib.pyplot as plt


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


def get_expected_time(used_modules:Dict[str, Any]) -> float:
    """
    Calculates how fast the moduels are likely to be executed [for comparison purposes, no unit intended].
    
    :param used_modules: Dictionary mapping module names to how often they are called.
    :returns: Expected execution time.
    """
    return sum(used_modules.values())


def get_module_confidences(explanans_collection:Dict[str, Dict]) -> Dict[str, float]:
    """
    Calculates for each module, what percent of the time it is used in the different code versions.
    
    :param explanans_collection: Explanans per code version.
    :returns: Dictionary of confidence per module.
    """
    modules = explanans_collection['']['Used Modules']
    confidences = {}
    
    for module in modules:
        num_occurences = sum([1 if module in explanans['Used Modules'] else 0 for explanans in explanans_collection])
        confidences[module] = num_occurences/len(explanans_collection.keys())
        
    return confidences


def gather_explanans(code:str, all_modules:List[str], used_modules:Dict[str,int]) -> Dict[str, Any]:
    """
    Gathers explanans for a given code snippet.
    
    :param code: Code snippet to generate explanans for.
    :param all_modules: All modules that could have been used in the code.
    :param used_modules: All modules that were used.
    :returns: Dictionary containing explanans related different aspects of the code.
    """
    explanans = {}
    explanans['Expected Time'] = get_expected_time(used_modules)
    explanans['Alternative Code'] = code
    explanans['Used Modules'] = used_modules
    explanans['Available Modules'] = all_modules
    return explanans


def summarize_explanans(explanans_collection:Dict[str, Dict]) -> Dict[str, Any]:
    """
    Summarizes previously collected explanans into a new Dictionary.
    
    :param explanans_collection: Dictionary of explanans collected per code version.
    :returns: Dictionary of summarized explanans.
    """
    summarized_explanans = {}
    summarized_explanans['Confidence'] = get_module_confidences(explanans_collection)
    
    base_time = explanans_collection['']['Time']
    summarized_explanans['Time Impact'] = {}
    for module in explanans_collection.keys():
        summarized_explanans['Time Impact'][module] = base_time - explanans_collection[module]['Expected Time']
    return summarized_explanans


def get_recommendation(summarized_explanans:Dict[str, Any], target:str) -> str:
    """
    Recommends which module to cut if any.
    
    :param summarized_explanans: Dictionary containing explanans for the code.
    :param target: String indicating which property to optimize.
    :returns: Name of the module recommendet to cut.
    """
    if target == 'Confidence':
        least_confident = np.argmin(summarized_explanans[target])
        return least_confident if summarized_explanans[target][least_confident] < 0.5 else ''
    return ''


def plot_explanations(summarized_explanations:Dict[str, Any]) -> None:
    """
    Plots the explanations.
    
    :param summarized_explanations: A python dictionary containing the summarized explanation.
    """
    # Sample data for heatmap
    heatmap_data = np.random.rand(10, 12)
    
    # Sample data for bar plot
    bar_data = summarized_explanations["Confidence"].values()
    
    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Create heatmap
    cax = ax1.imshow(heatmap_data, cmap='viridis')
    fig.colorbar(cax, ax=ax1)
    
    # Create bar plot
    ax2.bar(range(len(bar_data)), bar_data)
    
    # Add some labels for clarity
    ax1.set_title('Heatmap')
    ax2.set_title('Bar Plot')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plot
    plt.show()


def get_code_with_explanations(query:str, target:str) -> Tuple[str, Dict[str, Any]]:
    """
    Code generation variant that also generates explanations for the code.
    
    :param query: What query to generate code for.
    :param target: What you want to optimize for.
    :returns: Generated code, Dictionary containing explanans for the code, Module recommended to cut.
    """
    all_modules = []
    code_0 = viperGPT.get_code(query, module_list_out=all_modules)
    used_modules = identify_used_modules(code_0, all_modules)
    
    explanans_collection = {'': gather_explanans(code_0, all_modules, used_modules)}
    
    for module in used_modules.keys():
        reduced_modules = all_modules.copy()
        reduced_modules.remove(module)
        code_m = viperGPT.get_code(query, supressed_modules=[module])
        explanans_collection[module] = gather_explanans(code_m, reduced_modules, used_modules)

    summarized_explanans = summarize_explanans(explanans_collection)   
    recommendation = get_recommendation(summarized_explanans, target)
    return code_0, summarized_explanans, recommendation

plot_explanations({"Confidence":{"module1":0.6,"module2":0.4}, "Time Impact":{"module1":0.6,"module2":0.4}})