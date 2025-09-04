Structure:
- analysis_code contains code used for conducting the evaluation of results.
- viper-main contains the original ViperGPT-Code + extensions for generating explanations.
- The file InstallationIssues.txt contains instructions for fixing the ViperGPT installation. 
  Follow these guidelines to adapt the installation procedure presented in viper-main/README.md

Evaluation data is collected via the command "CONFIG_NAMES={insert_config_name} python main_batch.py" in the viper-main folder.
- Example: benchmarks/gqa executes evaluation on the gqa dataset
- The file configs/benchmarks/gqa.yaml can be edited to change the details of the evaluation plan specific to the gqa dataset.
- The file in configs/base_config.yaml can be edited to change the details of the explanations as well as basic ViperGPT settings.

Evaluation of the results is conducted via the analysis.py script. 
- When executing a window opens for selecting a folder. 
- Choose a folder that contains an results.csv as well as a explanations.json. 
- Then from the now displayed GUI pick any evaluation option. 
