import json
import pandas as pd
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
    confidences = data.get("Confidences", None)
    
    if confidences is None:
        raise KeyError('"Confidences" key not found in the JSON data.')
    
    if not isinstance(confidences, list):
        confidences = [confidences]

    sums = defaultdict(float)
    counts = defaultdict(int)

    for conf_dict in confidences:
        if isinstance(conf_dict, dict):
            for key, value in conf_dict.items():
                if isinstance(value, (int, float)):
                    sums[key] += value
                    counts[key] += 1

    means = {key: sums[key] / counts[key] for key in sums}
    print(means)
    
    # ###############
    #  Ties
    # ###############
    ties = data.get("Ties", None)

    if ties is None:
        raise KeyError('"Ties" key not found in the JSON data.')

    if not isinstance(ties, list):
        ties = [ties] 

    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))

    for outer_dict in ties:
        if isinstance(outer_dict, dict):
            for outer_key, inner_dict in outer_dict.items():
                if isinstance(inner_dict, dict):
                    for inner_key, value in inner_dict.items():
                        if isinstance(value, (int, float)):
                            sums[outer_key][inner_key] += value
                            counts[outer_key][inner_key] += 1

    means = {
        outer_key: {
            inner_key: sums[outer_key][inner_key] / counts[outer_key][inner_key]
            for inner_key in inner_keys
        }
        for outer_key, inner_keys in counts.items()
    }
    print(means)
    
    # Plot ties as heatmap
    df = pd.DataFrame.from_dict(means, orient='index')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Mean Ties from A to B")
    plt.xlabel("Module A")
    plt.ylabel("Module B")
    plt.tight_layout()
    plt.show()
    plt.savefig("TiesHeatmap.pdf")


if __name__ == "__main__":
    calculate_statistics()
