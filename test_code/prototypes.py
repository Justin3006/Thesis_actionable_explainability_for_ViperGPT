from typing import Dict, List

def gather_modules(code:str, exemptions:List[str]) -> List[str]:
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


def identify_used_modules(code:str, modules:List[str]) -> Dict[str, int]:
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


def count_leading_whitespace(text:str) -> int:
    count = 0
    for char in text:
        if char.isspace():
            count += 1
        else:
            break
    return count


def remove_function_definition(code: str, function_name: str) -> str:
    # Split the code into individual lines
    lines = code.split('\n')
    # Initialize a list to hold the lines of the updated code
    updated_lines = []
    # Flag to indicate whether we are inside the function to be removed
    inside_function = False
    # Flag to indicate whether we are inside Methods documentation
    inside_methods = False
    inside_definition = False
    reference_ident = 0

    for line in lines:
        # Check if the line contains the methods string
        if line.strip().startswith('Methods'):
            inside_methods = True
            reference_ident = count_leading_whitespace(line)
        
        # Check if the line contains a definition to remove
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

        # Check if the line contains the function definition we want to remove
        if line.strip().startswith(f'def {function_name}('):
            inside_function = True
            reference_ident = count_leading_whitespace(line)
            continue
        # If we are inside the function, skip adding the line to the updated code
        if inside_function:
            if count_leading_whitespace(line) <= reference_ident and len(line.strip()) > 0:
                # If we encounter an empty line, we might have reached the end of the function
                inside_function = False
            else:
                continue
        # Add the line to the updated code
        updated_lines.append(line)

    # Join the updated lines back into a single string
    return '\n'.join(updated_lines)


def remove_function_examples(code:str, function_name:str, execute_command:str) -> str:
    lines = code.split('\n')
    updated_lines = []

    example_start_ind = -1
    example_ident = -1
    function_found = False
    
    for ind, line in enumerate(lines):
        if example_start_ind != -1:
            if line.find(f'.{function_name}(') != -1 or line.find(f' {function_name}(') != -1:
                function_found = True
    
            if count_leading_whitespace(line) <= example_ident:
                if function_found:
                    while len(updated_lines) >= example_start_ind:
                        updated_lines.pop()
                example_start_ind = -1
                example_ident = -1
                function_found = False
                
        updated_lines.append(line)

        if line.find(f'def {execute_command}') != -1:
            example_start_ind = ind
            example_ident = count_leading_whitespace(line)

    return '\n'.join(updated_lines)


# Example usage:
input_code = '''
class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left, lower, right, upper : int
        An int describing the position of the (left/lower/right/upper) border of the crop's bounding box in the original image.

    Methods
    -------
    find(object_name: str)->List[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
        image matching the object_name.
    exists(object_name: str)->bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    """

    def __init__(self, image, left: int = None, lower: int = None, right: int = None, upper: int = None):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.
        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left, lower, right, upper : int
            An int describing the position of the (left/lower/right/upper) border of the crop's bounding box in the original image.
        """
        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
        else:
            self.cropped_image = image[:, lower:upper, left:right]
            self.left = left
            self.upper = upper
            self.right = right
            self.lower = lower

        self.width = self.cropped_image.shape[2]
        self.height = self.cropped_image.shape[1]

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

    def find(self, object_name: str) -> List[ImagePatch]:
        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        List[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop

        Examples
        --------
        >>> # return the foo
        >>> def execute_command(image) -> List[ImagePatch]:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patches = image_patch.find("foo")
        >>>     return foo_patches
        """
        return find_in_image(self.cropped_image, object_name)

    def exists(self, object_name: str) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.

        Examples
        -------
        >>> # Are there both foos and garply bars in the photo?
        >>> def execute_command(image)->str:
        >>>     image_patch = ImagePatch(image)
        >>>     is_foo = image_patch.exists("foo")
        >>>     is_garply_bar = image_patch.exists("garply bar")
        >>>     return bool_to_yesno(is_foo and is_garply_bar)
        """
        return len(find_in_image(self.cropped_image, object_name)) > 0

def combine():
    return 0
        
# Examples of how to use the API
# INSERT_QUERY_HERE
def execute_command(INSERT_TYPE_HERE):
    example = image_patch.find('stuff')
    returns example
    '''
    
result_code = '''# Some stuff here
# Example query here
def execute_command(INSERT_TYPE_HERE):
    example = image_patch.find('stuff')
    returns example
    
# Some more stuff here'''

modules = gather_modules(input_code, ['__init__', 'execute_command'])
print(modules)

used_modules = identify_used_modules(result_code, modules)
print(used_modules)

function_name = "combine"
updated_code = remove_function_definition(input_code, function_name)
print(updated_code)

updated_code = remove_function_examples(result_code, function_name, 'execute_command')
print(updated_code)
