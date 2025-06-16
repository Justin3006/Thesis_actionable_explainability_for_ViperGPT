Current idea for Explainability approach:
1. Generate code C_0 for some question via ViperGPT
2. Calculate and save score for C_0
2. Get list M of all modules used in C_0
3. For every module m in M:
	4. Generate new code C_m without m via ViperGPT 
	5. Calculate and save score for C_m
6. Choose m for which score is maximized (can be None)

For full automation:
7. Repeat from step 1 without that m

Score is calculated from different metrics like expected memory usage, expected runtime, etc.


TODO:
- Literature Review for actionable explainability methods applicable to ViperGPT
- Choose method to calculate score for this approach
- Implement and test approach (modeled after AdaCoder's test structure)
- Analyse and compare results to base-ViperGPT and the AdaCoder extension
