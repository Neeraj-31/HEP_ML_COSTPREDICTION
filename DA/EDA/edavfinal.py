import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Setup target directory
output_dir = 'edafinalplots'
os.makedirs(output_dir, exist_ok=True)

# 2. Load dataset
df = pd.read_csv('dam_ml_ready (1).csv')

# 3. Compute correlation matrix for numeric columns
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

# 4. Extract cost_overrun_pct column and sort from highest to lowest correlation
cost_corr = corr_matrix[['cost_overrun_pct']].sort_values(by='cost_overrun_pct', ascending=False)

# 5. Set up plot canvas dimension for clean single-strip rendering
plt.figure(figsize=(12, 24))

# 6. Draw the localized strip heatmap with clear text font size
sns.heatmap(
    cost_corr,
    annot=True,               # Show values on tiles
    fmt=".2f",                # Round to 2 decimal places
    cmap='coolwarm', 
    vmax=1.0, 
    vmin=-1.0, 
    center=0,
    linewidths=0.5,
    cbar_kws={"shrink": 0.7},
    annot_kws={"size": 12}    # Large and readable font on the tiles
)

# 7. Style ticks and title
plt.yticks(fontsize=12, rotation=0)
plt.xticks(fontsize=14)
plt.title('Correlation of Cost Overrun % with All Features', fontsize=18, pad=20)
plt.tight_layout()

# 8. Save the high-resolution isolated plot
plot_path = os.path.join(output_dir, 'cost_overrun_correlation.png')
plt.savefig(plot_path, dpi=150)
plt.close()

print(f"Isolated cost overrun correlation strip successfully saved to: {plot_path}")
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

# 4. Create a mask to keep only the lower triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# 5. Set up plot with a large canvas for high legibility
plt.figure(figsize=(40, 35))

# 6. Draw heatmap with cell annotations (values) and larger text sizes
sns.heatmap(
    corr_matrix, 
    mask=mask, 
    cmap='coolwarm', 
    vmax=1.0, 
    vmin=-1.0, 
    center=0,
    square=True, 
    linewidths=0.5, 
    cbar_kws={"shrink": 0.5}, 
    annot=True,               # Enables correlation numbers on tiles
    fmt=".2f",                # Formats numbers to 2 decimal places
    annot_kws={"size": 10}    # Enlarges font size of numbers inside tiles
)

# 7. Adjust font size for axis labels and title
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14, rotation=0)
plt.title('Lower Triangle Correlation Matrix (Annotated)', fontsize=28, pad=30)
plt.tight_layout()

# 8. Save the high-resolution plot directly to the directory
plot_path = os.path.join(output_dir, 'correlation_matrix_annotated.png')
plt.savefig(plot_path, dpi=150)
plt.close()