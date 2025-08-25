from importlib import metadata
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


def get_module_confidences(metadata_collection:List[Dict[str, Dict]], ties:Dict[str, Dict]) -> Dict[str, float]:
    """
    Calculates for each module, what percent of the time it is used in the different code versions.
    
    :param metadata_collection: Metadata per code version.
    :param ties: Dict containing ties per module.
    :returns: Dictionary of confidence per module.
    """
    modules = metadata_collection[0]['']['Available Modules']
    confidences = {module:(1 if module in metadata_collection[0] else 0) for module in modules}
    
    for cycle in range(len(metadata_collection)):
        for module in modules:
            num_occurences = sum([1 if module in metadata['Used Modules'] else 0 for other_module, metadata in metadata_collection[cycle].items()])
            confidences[module] += num_occurences
            
    for module in confidences: 
        confidences[module] /= (len(metadata_collection) * (len(metadata_collection[cycle]) - (1 if module in metadata_collection[cycle] else 0)) + 1)

    return confidences


def get_module_ties(metadata_collection:List[Dict[str, Dict]]) -> Dict[str, Dict]:
    """
    Calculates for each module, what percent of the time it is used with each other module.
    
    :param metadata_collection: Metadata per code version.
    :returns: Dictionary of module ties per module.
    """
    modules = metadata_collection[0]['']['Available Modules']
    ties = {module:{other_module:0 for other_module in modules} for module in modules}
    
    for module in modules:
        num_occurences = 1 if module in metadata_collection[0] else 0
        for cycle in range(len(metadata_collection)):
            num_occurences += sum([1 if module in metadata['Used Modules'] else 0 for m, metadata in metadata_collection[cycle].items()])
            
        if num_occurences > 0:
            for other_module in modules:
                num_sim_occurences = 1 if module in metadata_collection[0] and other_module in metadata_collection[0] else 0
                for cycle in range(len(metadata_collection)):
                    num_sim_occurences += sum([1 if module in metadata['Used Modules'] and other_module in metadata['Used Modules'] else 0 for m, metadata in metadata_collection[cycle].items()])
                ties[module][other_module] += num_sim_occurences/num_occurences
                
    return ties


def gather_metadata(code:str, all_modules:List[str], used_modules:Dict[str,int]) -> Dict[str, Any]:
    """
    Gathers metadata for a given code snippet.
    
    :param code: Code snippet to generate metadata for.
    :param all_modules: All modules that could have been used in the code.
    :param used_modules: All modules that were used.
    :returns: Dictionary containing metadata related different aspects of the code.
    """
    metadata = {}
    metadata['Alternative Code'] = code
    metadata['Used Modules'] = used_modules
    metadata['Available Modules'] = all_modules
    return metadata


def generate_explanation(metadata_collection:List[Dict[str, Dict]]) -> Dict[str, Any]:
    """
    Generates an explanation from previously collected metadata.
    
    :param metadata_collection: List of dictionary of metadata collected per code version per cycle.
    :returns: Dictionary containing explanation elements.
    """
    explanation = {}
    ties = get_module_ties(metadata_collection)
    confidences = get_module_confidences(metadata_collection, ties)
    
    for module in metadata_collection[0]['']['Available Modules']:
        explanation_for_module = {}
        explanation_for_module['Confidence'] = confidences[module] if module != '' else 1
        explanation_for_module['Ties'] = ties[module] if module != '' else {}
        if module in metadata_collection[0]:
            explanation_for_module['Alternative Code'] = [metadata_collection[cycle][module]['Alternative Code'] for cycle in range(len(metadata_collection))]
        else:
            explanation_for_module['Alternative Code'] = ['' for cycle in range(len(metadata_collection))]
        explanation[module] = explanation_for_module

    return explanation


def get_recommendation(explanation:Dict[str, Any], threshold:float) -> List[str]:
    """
    Recommends which module to cut if any.
    
    :param explanation: Dictionary containing explanation for the code.
    :param threshold: Confidence threshold below which to cut modules.
    :returns: Name of the module recommendet to cut.
    """
    used = [module for module, explanation_for_module in explanation.items() if explanation_for_module['Alternative Code'][0] != '']
    not_used = [module for module in explanation if module != '' and module not in used]
    threshold = np.max([explanation[module]['Confidence'] for module in not_used])
    below_threshold = [module for module in used if explanation[module]['Confidence'] < threshold]
    return below_threshold


def save_explanation(explanation:Dict[str, Any], filename: str = 'explanation') -> None:
    """
    Save explanation in a json file.
    
    :param explanation: Explanation to save.
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

    for key, value in explanation.items():
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
#    :returns: Generated code, Dictionary containing explanation for the code, Module recommended to cut.
#    """
#    all_modules = []
#    code_0 = viperGPT.get_code(query, module_list_out=all_modules)
#    used_modules = identify_used_modules(code_0, all_modules)
#    
#    metadata_collection = {'': gather_metadata(code_0, all_modules, used_modules)}
#    
#    for module in used_modules.keys():
#        reduced_modules = all_modules.copy()
#        reduced_modules.remove(module)
#        code_m = viperGPT.get_code(query, supressed_modules=[module])
#        metadata_collection[module] = gather_metadata(code_m, reduced_modules, used_modules)
#
#    explanation = generate_explanation(metadata_collection)   
#    recommendation = get_recommendation(explanation, target)
#    return code_0, explanation, recommendation
