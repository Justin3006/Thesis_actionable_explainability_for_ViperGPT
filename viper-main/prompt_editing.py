from typing import List

def gather_modules(code:str, exemptions:List[str]) -> List[str]:
    """
    Finds all defined modules in a code snippet.
    
    :param code: Code to find modules in.
    :param exemptions: Names of functions to not count as modules.
    :returns: List of module names.
    """
    modules = []
    lines = code.split('\n')
    
    for line in lines:
        if line.strip().startswith('def '):
            start = line.find('def ') + 4
            end = line.find('(')
            name = line[start:end]
            if name not in exemptions:
                modules.append(line[start:end])
    return modules


def count_leading_whitespace(text:str) -> int:
    """
    Count the number of leading whitespaces in a line of text (i.e. the identation).
    
    :param text: Line of text to count keading whitespaces of.
    :returns: Number of leading whitespaces.
    """
    count = 0
    for char in text:
        if char.isspace():
            count += 1
        else:
            break
    return count


def remove_function_definition(code: str, function_name: str) -> str:
    """
    Removes the function definition with the given name from a code snippet.
    
    :param code: Code to edit.
    :param function_name: Name of the function to remove definition of.
    :returns: Edited code.
    """
    lines = code.split('\n')  # Old/new code.
    updated_lines = []
    
    inside_function = False  # Flags for controlling behaviour.
    inside_methods = False
    inside_definition = False
    reference_ident = 0  # Distinguish code blocks by identation.

    # Go through code line by line.
    for line in lines:
        # Check if in documentation.
        if line.strip().startswith('Methods'):
            inside_methods = True
            reference_ident = count_leading_whitespace(line)
        
        # If in documentation, don't add parts concerning function_name to new code.
        if inside_methods:
            if line.strip().startswith(function_name):
                inside_definition = True
                continue
            if inside_definition:
                if count_leading_whitespace(line) > reference_ident:
                    continue
                else:
                    inside_definition = False
            if line.strip().startswith('"""'): 
                inside_methods = False

        # Check if in function definition.
        if line.strip().startswith(f'def {function_name}('):
            inside_function = True
            reference_ident = count_leading_whitespace(line)
            continue
        
        # If in function definition, don't add line to new code.
        if inside_function:
            if count_leading_whitespace(line) <= reference_ident and len(line.strip()) > 0:
                inside_function = False
            else:
                continue
            
        # Retain line in update.
        updated_lines.append(line)

    return '\n'.join(updated_lines)


def remove_function_examples(code:str, function_name:str, execute_command:str) -> str:
    """
    Removes all code examples containing the function of the specified name from a code snippet.
    
    :param code: Code to edit.
    :param function_name: Name of the function to remove examples of.
    :param execute_command: signifier of example start.
    :returns: Edited code.
    """
    lines = code.split('\n')  # Old/new code.
    updated_lines = []

    example_start_ind = -1  # Identify code blocks by identation and starting line. 
    example_ident = -1
    function_found = False
    
    # Go through code line by line.
    for ind, line in enumerate(lines):
        original_line = line
        
        temp_ind = line.find(">>>")
        if temp_ind != -1:
            line = line[:temp_ind] + line[temp_ind+3:]

        # If in function, look for function_name and remove from updated code if found.
        if example_start_ind != -1:
            if line.find(f'.{function_name}(') != -1 or line.find(f' {function_name}(') != -1:
                function_found = True
    
            if count_leading_whitespace(line) <= example_ident:
                if function_found:
                    for i in range(ind - example_start_ind + 2):
                        updated_lines.pop()
                example_start_ind = -1
                example_ident = -1
                function_found = False
                
        # Retain line in new code.
        updated_lines.append(original_line)

        # Mark beginning of new function, if necessary.
        if line.find(f'def {execute_command}') != -1:
            example_start_ind = ind
            example_ident = count_leading_whitespace(line)

    return '\n'.join(updated_lines)
