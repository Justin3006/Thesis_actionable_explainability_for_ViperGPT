TODO:
- validate fidelity and distinctiveness of Confidence and Ties via what we already know about modules
  - RefCOCO should have 'find' at roughly 100% Confidence, way higher than any other module (based on examples)
  - GQA should have 'bool_to_yesno' have very high Ties with 'verify_property', 'exist', etc. (requires boolean input, these give boolean answers)
- Test approach on GQA, RefCOCO and OK-VQA
  - Analyze correlation between confidence and accuracy 
  - Test for accuracy improvement via module selection
  - Get confidence and ties statistics for different modules
- Verbalize approach in chapter 4
- Add more examples to chapter 3 

Note:
analysis_code contains code used for conducting the evaluation.
viper-main contains the original ViperGPT-Code + extensions for generating explanations
