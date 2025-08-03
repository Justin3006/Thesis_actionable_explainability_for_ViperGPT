Current idea for Explainability approach:
1. Generate code C_0 for some question via ViperGPT
2. Calculate and save score for C_0
3. Get list M of all modules used in C_0
4. For every module m in M:
	5. Generate new code C_m without m via ViperGPT 
	6. Calculate and save score for C_m
7. Choose m for which score is maximized (can be None)

For full automation:
8. Repeat from step 1 without that m

Score is calculated from ties between modules and confidence.


Rough thesis outline:
- Literature Review for actionable explainability methods applicable to ViperGPT
- Choose method to calculate score for this approach
- Implement and test approach
- Analyse and compare results to base-ViperGPT and the AdaCoder extension


TODO:
- Test approach on GQA, RefCOCO and OK-VQA
  - Test for accuracy improvement
  - Get confidence and ties statistics for different modules
- Verbalize approach
- fine-tune first three chapters of the thesis 
