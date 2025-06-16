Current idea for Explainability approach:
1. Generate code C_0 for some question via ViperGPT
2. Measure and record given metrics for C_0
2. Create a list M of all modules used in C_0
3. For every module m in M:
	4. Generate new code C_m without m via ViperGPT 
	5. Measure and record given metrics for C_m
6. From all recorded metrics, determine which m should be removed (if any)

For full automation:
7. Repeat from step 1 without respective m

Metrics could include expected memory usage, expected runtime, etc.


TODO:
- Literature Review for explainability methods applicable to ViperGPT
- Choose metrics to apply for this approach
- Implement and test approach
- Analyse and compare results to base-ViperGPT and its AdaCoder extension
