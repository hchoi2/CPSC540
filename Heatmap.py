import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Extract the results into a DataFrame
results_df = pd.read_excel('Cat_heat.xlsx')

# Extract relevant columns for the heatmap
heatmap_data = results_df.pivot(index='param_l2_leaf_reg', columns='param_max_depth', values='mean_test_score')

custom_cmap = ListedColormap(sns.color_palette("Greens", as_cmap=True)(range(10, 256)))
# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap=custom_cmap, fmt=".3f", cbar_kws={'label': 'Weighted F1 Score'})
plt.title('GridSearchCV Results Heatmap')
plt.xlabel('max_depth')
plt.ylabel('l2_leaf_reg')
plt.show()