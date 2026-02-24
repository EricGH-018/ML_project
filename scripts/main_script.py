# PACKAGES TO FILTER AND STORE DATA
import pandas as pd 
import numpy as np 
import missingno as msno 
import janitor # pyjanitor

# PACKAGES TO VISUALIZE DATA
from skimpy import skim 
import matplotlib.pyplot as plt 
import seaborn as sns

# PACKAGES TO APPLY ML METHODS
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load data from subjects into a single DataFrame (using a list comprehension to avoid innecessary storage).
dataset = pd.concat([pd.read_csv(f'subject10{i}.dat', sep=r'\s+', header=None) # read every .dat file, without the header. 
                    for i in range(1,10)], ignore_index=True) # iterate through every subject. 

# Rename columns in the DataFrame
col_names = ["timestamp","activity_ID","heart_rate","Temp_hand","3D_acc1_IMU_hand_scale1",
		"3D_acc2_IMU_hand_scale1","3D_acc3_IMU_hand_scale1","3D_acc1_IMU_hand_scale2",
                "3D_acc2_IMU_hand_scale2","3D_acc3_IMU_hand_scale2", "3D_gyroscope1_IMU_hand",
		"3D_gyroscope2_IMU_hand", "3D_gyroscope3_IMU_hand", "3D_magnetometer1_IMU_hand",
		"3D_magnetometer2_IMU_hand", "3D_magnetometer3_IMU_hand", "orientation1_IMU_hand",
		"orientation2_IMU_hand", "orientation3_IMU_hand", "orientation4_IMU_hand","Temp_chest","3D_acc1_IMU_chest_scale1",
                "3D_acc2_IMU_chest_scale1","3D_acc3_IMU_chest_scale1","3D_acc1_IMU_chest_scale2",
                "3D_acc2_IMU_chest_scale2","3D_acc3_IMU_chest_scale2", "3D_gyroscope1_IMU_chest",
                "3D_gyroscope2_IMU_chest", "3D_gyroscope3_IMU_chest", "3D_magnetometer1_IMU_chest",
                "3D_magnetometer2_IMU_chest", "3D_magnetometer3_IMU_chest", "orientation1_IMU_chest",
                "orientation2_IMU_chest", "orientation3_IMU_chest", "orientation4_IMU_chest","Temp_ankle","3D_acc1_IMU_ankle_scale1",
                "3D_acc2_IMU_ankle_scale1","3D_acc3_IMU_ankle_scale1","3D_acc1_IMU_ankle_scale2",
                "3D_acc2_IMU_ankle_scale2","3D_acc3_IMU_ankle_scale2", "3D_gyroscope1_IMU_ankle",
                "3D_gyroscope2_IMU_ankle", "3D_gyroscope3_IMU_ankle", "3D_magnetometer1_IMU_ankle",
                "3D_magnetometer2_IMU_ankle", "3D_magnetometer3_IMU_ankle", "orientation1_IMU_ankle",
                "orientation2_IMU_ankle", "orientation3_IMU_ankle", "orientation4_IMU_ankle"
]

dataset.columns = col_names # method to apply names to columns using a list.
dataset = dataset[[col for col in dataset.columns if 'orientation' not in col]] # remove orientation data (invalid in this study).

valid_values = [1, 2, 3, 4, 5, 6]
dataset = dataset[dataset['activity_ID'].isin(valid_values)] # keep specific activities only

# Show in the terminal a first section of information: dataset overview.
print("\n--- DATASET DIMENSIONS: (rows, columns) ---")
print("Dimensions of the DataFrame: ",dataset.shape)
print("\n--- OVERVIEW OF THE DATASET (first 5 rows) ---")
print(dataset.head())

# Inspect the content to finda NA values (no information).
na_summary = ( 
		dataset.isna().sum() # find and sum all NA cells. 
		.reset_index() 
		.rename(columns={"index": "col", 0: "na_count"}) # rename columns in the final output.
)

na_summary["na_pct"] = na_summary["na_count"] / len(dataset) # give a measure of relative quantity as a new column.

# Show in the terminal a second section of information: NaN exploratory analysis.
print("\n--- NaN VALUES SUMMARY PER VARIABLE ---")
print(na_summary)

# Generate a plot to visualize distribution of NaN values.
print("\n --- ELABORATE PLOTS TO VISUALIZE NaN VALUES ---")
msno.matrix(dataset)
plt.savefig("na_distribution.png", bbox_inches='tight')
plt.close()
print(f"NaN distribution plot saved successfully as na_distribution.png")

msno.bar(dataset)
plt.savefig("na_chart.png", bbox_inches='tight')
plt.close()
print(f"NaN chart plot saved successfully as na_chart.png")

# Remove rows with NaN values
dataset = dataset.dropna(axis=0, how='any') # 'how=any' to clear all rows with at least 1 missing value (at least for heart_rate).

# Show in the terminal a third section of information: dataset after filtering NaN values.
print("\n--- DATASET DIMENSIONS (after filtering for NaN values): (rows, columns) ---")
print("Dimensions of the DataFrame: ",dataset.shape)

na_summary = (
                dataset.isna().sum() # find and sum all NA cells.
                .reset_index()
                .rename(columns={"index": "col", 0: "na_count"}) # rename columns in the final output.
)

na_summary["na_pct"] = na_summary["na_count"] / len(dataset) # give a measure of relative quantity as a new column.

print("\n--- NaN VALUES SUMMARY PER VARIABLE ---\n")
print(na_summary)

# Show in the temrinal a fourth section of information: outliers identification + distribution of the variables.
print("\n--- SAVE AN IMAGE TO EVALUATE DISTRIBUTIONS ---")

def plot_distrib(dataset):
        variables = dataset.columns.tolist()

        # Set up a matrix to represent each plot.
        fig, axes = plt.subplots(6, 7, figsize=(35, 20))
        axes = axes.flatten()

        for i, j in enumerate(variables):
                sns.histplot(dataset[j], kde=True, ax=axes[i], color='teal')
                axes[i].set_title(f'Distribution of {j}',fontsize=10)
                axes[i].set_xlabel('')

        plt.tight_layout()
        plt.savefig(f'Distribution_variables.png')
        plt.close()
        print(f'Saved distribution for variables as: Distribution_variables.png')

plot_distrib(dataset)

print("\n--- EVALUATE OUTLIERS ---")

def outliers_id(dataset): # function to identify outliers in data using Z-score:

    dataset = dataset.reset_index(drop=True)

    variables = dataset.columns.tolist()

    fig, axes = plt.subplots(6, 7, figsize=(35, 20), sharey=False)
    axes = axes.flatten()

    outlier_id = [] 

    for num, element in enumerate(variables):
        # Extract only a specific column to identify outliers in it.
        col = dataset[element]
    
        # Calculate Z-score manually as (col - mean) / std
        z = (col - col.mean()) / col.std()

        # Identify indices where absolute z-score > 3 (arbitrary threshold).
        outliers_z_idx = dataset[z.abs() > 3].index 

        for out in outliers_z_idx: 
            if out not in outlier_id: 
                outlier_id.append(out) 

        print(f"Number of outliers in {element}: {len(outliers_z_idx)}")

        # Display the outlier rows
        outlier_data = dataset.loc[outliers_z_idx] 
        if not outlier_data.empty:
            print(outlier_data)

        # Access the specific subplot in the 2D array
        ax = axes[num]
        ax.boxplot(col.dropna(), vert=True, patch_artist=True, boxprops=dict(facecolor='skyblue', color='blue'), showfliers=False, flierprops=dict(marker='o', markerfacecolor='red', markersize=8))
        ax.scatter([1] * len(outlier_data), outlier_data[element], color='red', edgecolors="black", marker='o', label='Z-outlier', zorder=3, s=50)

        ax.set_xlabel(element)

    # Save the plot with boxplots and the corresponding outliers marked with red dots.
    plt.tight_layout()
    plt.savefig(f'Outliers_variables.png')
    plt.close()
    print(f'\nSaved boxplots for variables as: Outliers_variables.png')
    dataset = dataset.drop(outlier_id)

    print(f'Total number of rows removed (unique outliers): {len(outlier_id)}')
    return dataset

dataset = outliers_id(dataset)

# Show in the terminal a fifth section of information: correlation matrix.
print("\n--- CREATE A CORRELATION MATRIX TO KNOW IF ANY PAIR OF VARIABLES ARE ASSOCIATED BETWEEN THEM ---")

plt.figure(figsize=(30, 25))

corr = dataset.corr() # calculate correlation for variables in the dataset. 
sns.heatmap(corr, fmt=".2f", annot=True, cmap='coolwarm', linewidths=0.5) # create a heatmap. 
plt.title("Feature Correlation Matrix") # add a title.

plt.tight_layout()
plt.savefig(f'Correlation_matrix.png')
plt.close()
print("Saved an image for correlation matrix as: Correlation_matrix.png")

# Output the new dimensions of the dataset, after filtering. 
dataset = dataset.loc[:, ~dataset.columns.str.contains('scale2')] # symbol ~ means NOT, basically avoid those columns.
print("\n--- DATASET DIMENSIONS (after all filtering steps): (rows, columns) ---")
print("Dimensions of the DataFrame: ",dataset.shape)

# Show a sixth section of information: clusterization with PCA depending on the activity ID.
print("\n--- PERFORM A PCA TO EVALUATE THE DISTINCTION BETWEEN CLUSTERS ---") 

data_features=dataset.iloc[:,2:] # avoid columns for timestamp and activity ID. 
scaler = StandardScaler() # import a scaling method.
data_features_scaled = scaler.fit_transform(data_features) # scale data.

# Load an apply PCA on scaled data.
model = PCA(n_components=2)
Xlowdim = model.fit_transform(data_features_scaled) 

# Generate a plot to visualize the results of PCA.
plt.figure(figsize=(15, 15))

scatter = plt.scatter(Xlowdim[:, 0], Xlowdim[:, 1], c=dataset["activity_ID"].tolist(), cmap='viridis', alpha=0.7) # dots colored as classification in activity_ID column

plt.xlabel("PC1", fontweight='bold')
plt.ylabel("PC2", fontweight='bold')
plt.title("PCA with activity ID clusters", fontsize=14)

# Show a gradient at the left to identify colors for every activity.
plt.colorbar(scatter, label='Activity ID')

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('PCA.png')
print("\nSaved an image for the result of PCA using the activity ID as labels (PCA.png)\n")

# Show in the terminal a datframe for the contribution of variables to PC1 and PC2.
loadings = pd.DataFrame(
    model.components_.T, 
    columns=['PC1', 'PC2'], 
    index=data_features.columns.tolist()
)

print("Values affecting PC1")
print(loadings['PC1'].sort_values(ascending=False))
print("\nValues affecting PC2")
print(loadings['PC2'].sort_values(ascending=False))

# Show a seventh section of information: application of a supervised machine learning method, random forests. Evaluate results for the category column in the original dataset.
print("\n--- APPLY RANDOM FORESTS ALGORITHM ON DATA ---")
proportions = dataset['activity_ID'].value_counts() / len(dataset['activity_ID']) # calculate proportion of every activity.

print("Proportions of activity_ID")
print(proportions) 

# Divide the dataset in variables used to predict (X) and the predictions (Y)
X = dataset.drop(columns=['activity_ID', 'timestamp'])
y = dataset['activity_ID']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=False) 
axes = axes.flatten()

# Evaluate results for the category column in train and test datasets.
proportions_train = y_train.value_counts() / len(y_train) 
proportions_test = y_test.value_counts() / len(y_test)

# Create pie chart for training and test datasets.
prop_train = proportions_train.sort_index()
prop_test = proportions_test.sort_index()

activity_colors = ['#008080', '#87CEEB', '#0000FF', '#20B2AA', '#008B8B', '#B0E0E6']

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Training data.
prop_train.plot.pie(
    ax=axes[0], 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=activity_colors, 
    pctdistance=0.85, # Moves percentages outward slightly
    ylabel=''
)
axes[0].set_title('Training Set Distribution', fontsize=12, fontweight='bold')

# Test data.
prop_test.plot.pie(
    ax=axes[1], 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=activity_colors, 
    pctdistance=0.85, 
    ylabel=''
)
axes[1].set_title('Test Set Distribution', fontsize=12, fontweight='bold')

lines, labels = axes[0].get_legend_handles_labels()
fig.legend(lines, prop_train.index, title="Activity ID", loc="center right", borderaxespad=0.1)

# Adjust layout to include a legend.
plt.subplots_adjust(right=0.85) 
plt.savefig('Proportions.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nSaved Proportions.png as a pie chart for information on training and test datasets.\n")

# Scale data to avoid unequal contribution.
scaler = StandardScaler() 

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Import the model and apply it to training data. 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 to use all CPUs, n_estimators=100 as an arbitrary measure.
rf_model.fit(X_train_scaled, y_train) # train the model

y_pred = rf_model.predict(X_test_scaled) # use the model to make predictions on test data

print("Random Forest Classification Report")
report = classification_report(y_test, y_pred) # give a series of metrics on the performance of the model (precision, recall, f1-score).
print(report)

accuracy = accuracy_score(y_test, y_pred) # extract accuracy for the model.
print(f"Overall Accuracy: {accuracy:.2%}")

print("\n--- GENERATE FIGURES TO EVALUATE PERFORMANCE OF THE MODEL ---")

# Confusion Matrix elaboration:
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred) # store the confusion matrix in an object.
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',  # represent information in a heatmap.
            xticklabels=rf_model.classes_, 
            yticklabels=rf_model.classes_)

plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})', fontsize=14)
plt.xlabel('Predicted Activity ID', fontsize=12)
plt.ylabel('True Activity ID', fontsize=12)
plt.tight_layout()

plt.savefig('RF_Confusion_Matrix.png', dpi=300)
print("\nSaved an image for the Confusion Matrix (RF_Confusion_Matrix.png)\n")
plt.close()

# Feature importance:
importances = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=True) # contribution of every variable to the model (feature_importances_)

# Save a figure as a barplot. 
plt.figure(figsize=(10, 12))
importances.plot(kind='barh', color='teal')
plt.title('Feature Importance for Activity Classification', fontsize=14)
plt.xlabel('Importance Score', fontsize=12)
plt.tight_layout()

plt.savefig('RF_Feature_Importance.png', dpi=300)
print("\nSaved an image for the Feature Importance (RF_Feature_Importance.png)\n")
plt.close()

print("\nProcessing completed. Analysis ended")
print("--------------------------------------\n")
