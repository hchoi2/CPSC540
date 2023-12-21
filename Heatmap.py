import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Extract the results into a DataFrame
results_df = pd.read_excel('LGBM_heat.xlsx')

# Extract relevant columns for the heatmap
heatmap_data = results_df.pivot(index='param_num_leaves', columns='param_feature_fraction_bynode', values='mean_test_score')

custom_cmap = ListedColormap(sns.color_palette("Greens", as_cmap=True)(range(10, 256)))
# Plot the heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(heatmap_data, annot=True, cmap=custom_cmap, fmt=".3f", cbar_kws={'label': 'Weighted F1 Score'})
plt.title('LightGBM Heatmap')
plt.xlabel('feature_fraction_bynode')
plt.ylabel('num_leaves')
plt.show()