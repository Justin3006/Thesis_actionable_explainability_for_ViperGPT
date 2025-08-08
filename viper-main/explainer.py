from typing import Dict, List, Any, Tuple
#import main_simple_lib as viperGPT
import numpy as np
import json
from configs import config
import pathlib
import os


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


def get_module_confidences(explanans_collection:Dict[str, Dict], ties:Dict[str, Dict]) -> Dict[str, float]:
    """
    Calculates for each module, what percent of the time it is used in the different code versions.
    
    :param explanans_collection: Explanans per code version.
    :param ties: Dict containing ties per module.
    :returns: Dictionary of confidence per module.
    """
    modules = explanans_collection['']['Available Modules']
    confidences = {}
    
    for module in modules:
        num_occurences = sum([1 if module in explanans['Used Modules'] or other_module != '' and ties[module][other_module] == 1 else 0 for other_module, explanans in explanans_collection.items()])
        confidences[module] = num_occurences/len(explanans_collection)
        
    return confidences


def get_module_ties(explanans_collection:Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Calculates for each module, what percent of the time it is used with each other module.
    
    :param explanans_collection: Explanans per code version.
    :returns: Dictionary of module ties per module.
    """
    modules = explanans_collection['']['Available Modules']
    ties = {}
    
    for module in modules:
        ties_m = {}
        num_occurences = sum([1 if module in explanans['Used Modules'] else 0 for m, explanans in explanans_collection.items()])
        if num_occurences > 0:
            for other_module in modules:
                num_sim_occurences = sum([1 if module in explanans['Used Modules'] and other_module in explanans['Used Modules'] else 0 for m, explanans in explanans_collection.items()])
                ties_m[other_module] = num_sim_occurences/num_occurences
        else:
            ties_m = {other_module:0 for other_module in modules}
        
        ties[module] = ties_m

    return ties


def gather_explanans(code:str, all_modules:List[str], used_modules:Dict[str,int]) -> Dict[str, Any]:
    """
    Gathers explanans for a given code snippet.
    
    :param code: Code snippet to generate explanans for.
    :param all_modules: All modules that could have been used in the code.
    :param used_modules: All modules that were used.
    :returns: Dictionary containing explanans related different aspects of the code.
    """
    explanans = {}
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
    ties = get_module_ties(explanans_collection)
    confidences = get_module_confidences(explanans_collection, ties)
    
    for module in explanans_collection['']['Available Modules']:
        summarized_explanans[module]['Confidence'] = confidences[module] if module != '' else 1
        summarized_explanans[module]['Ties'] = ties[module] if module != '' else {}
        if module in explanans_collection['']['Used Modules']:
            summarized_explanans[module]['Alternative Code'] = explanans_collection[module]['Alternative Code']
        else:
            summarized_explanans[module]['Alternative Code'] = ''

    return summarized_explanans


def get_recommendation(summarized_explanans:Dict[str, Any], threshold:float) -> List[str]:
    """
    Recommends which module to cut if any.
    
    :param summarized_explanans: Dictionary containing explanans for the code.
    :param threshold: Confidence threshold below which to cut modules.
    :returns: Name of the module recommendet to cut.
    """
    used = identify_used_modules(summarized_explanans['']['Alternative Code'])
    not_used = [module for module in summarized_explanans if module != '' and module not in used]
    threshold = np.max([summarized_explanans[module]['Confidence'] for module in not_used])
    below_threshold = [module for module in used if summarized_explanans[module]['Confidence'] < threshold]
    return below_threshold


def save_explanation(summarized_explanans:Dict[str, Any], filename: str = 'explanans') -> None:
    """
    Save explanation in a json file.
    
    :param summarized_explanans: Explanation to save.
    """
    results_dir = pathlib.Path(config['results_dir'])
    results_file = results_dir/filename
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    for key, value in summarized_explanans.items():
        if key in existing_data:
            if not isinstance(existing_data[key], list):
                existing_data[key] = [existing_data[key]]
            if isinstance(value, list):
                existing_data[key].extend(value)
            else:
                existing_data[key].append(value)
        else:
            existing_data[key] = value

    with open(results_file, 'w') as f:
        json.dump(existing_data, f, indent=4)


#def get_code_with_explanations(query:str, target:str) -> Tuple[str, Dict[str, Any]]:
#    """
#    Code generation variant that also generates explanations for the code.
#    
#    :param query: What query to generate code for.
#    :param target: What you want to optimize for.
#    :returns: Generated code, Dictionary containing explanans for the code, Module recommended to cut.
#    """
#    all_modules = []
#    code_0 = viperGPT.get_code(query, module_list_out=all_modules)
#    used_modules = identify_used_modules(code_0, all_modules)
#    
#    explanans_collection = {'': gather_explanans(code_0, all_modules, used_modules)}
#    
#    for module in used_modules.keys():
#        reduced_modules = all_modules.copy()
#        reduced_modules.remove(module)
#        code_m = viperGPT.get_code(query, supressed_modules=[module])
#        explanans_collection[module] = gather_explanans(code_m, reduced_modules, used_modules)
#
#    summarized_explanans = summarize_explanans(explanans_collection)   
#    recommendation = get_recommendation(summarized_explanans, target)
#    return code_0, summarized_explanans, recommendation
