StatTick 

StatTick is a lightweight, LLM-powered command-line toolkit designed to bridge the gap between natural language and mathematical computing. Whether you are a seasoned researcher or a total beginner, StatTick allows you to perform complex statistical analysis by simply asking questions.

🎯 What is StatTick?

StatTick blends traditional statistical rigor with the flexibility of Large Language Models (LLMs). Instead of writing complex Python scripts or manual Excel formulas, you can interact with your data using "Statistical Intent."

The tool interprets your prompts, maps them to mathematical functions, and returns high-fidelity outputs instantly.

✨ Key Features

1. Natural Language Interface: Ask questions like "Run a linear regression between height and weight" or "What is the correlation between age and income?"

2. Robust Statistical Outputs: Get real-time R² values, regression coefficients, p-values, and model summaries.

3. Comprehensive Toolkit: Supports basic descriptive stats, multi-variable regression, and rank correlation (Spearman/Kendall).

4.  Seamless Data Handling: Load and clean .csv or .xlsx datasets with a single line of text.

5. Beginner Friendly: Built specifically for those who want the power of data science without the steep coding curve.

Usage Examples

Once you launch the StatTick CLI, you can start exploring your data:

1. Loading Data:

"Load my data from 'survey_results.csv' and remove any rows with missing values."

2. Simple Analysis:

"What is the mean and standard deviation of the 'test_scores' column?"

3. Regression Analysis:

"Perform a multi-variable regression where 'Salary' is the dependent variable and 'Experience' and 'Education_Level' are predictors."

4. Correlation:

"Check the rank correlation between 'Customer_Satisfaction' and 'Wait_Time'."

🛠️ Built With

LLM Integration: Interprets natural language into executable statistical logic.

Pandas: For high-performance data manipulation and cleaning.

Statsmodels / Scipy: For accurate mathematical computation.

Typer/Click: For the smooth command-line interface experience.

