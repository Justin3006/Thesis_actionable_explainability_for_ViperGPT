from typing import Dict, List, Any, Tuple
import main_simple_lib as viperGPT


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


def get_code_with_explanations(query:str) -> Tuple[str, Dict[str, Any]]:
    """
    Code generation variant that also generates explanations for the code.
    
    :param query: What query to generate code for.
    :returns: Generated code, Dictionary containing explanans for the code.
    """
    all_modules = []
    code_0 = viperGPT.get_code(query, all_modules) # TODO: add another parameter
    used_modules = identify_used_modules(code, all_modules)
    # TODO: Record whatever data you need.
    for module in used_modules.keys():
        reduced_modules = all_modules.copy()
        reduced_modules.remove(module)
        code = viperGPT.get_code(query, reduced_modules) # TODO: add another parameter
        # TODO: Record whatever data you need.
    
    explanans = {}
    # TODO: Add results to explanans. 
    return code_0, explanans
    
