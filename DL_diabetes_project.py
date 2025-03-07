import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import layers, models
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv('Diabetes Simple Diagnosis.csv')
df.drop(df.columns[0], axis=1, inplace=True)

# Display basic information
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

## Calculate summary statistics
summary = df.describe()
summary = summary.round(4)
# Convert the summary statistics to a DataFrame for easy plotting
summary_df = summary.reset_index()
# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 5))
# Hide the axes
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)
# Create a table
table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, cellLoc='center', loc='center')
# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
# Show the table
plt.show()

# diabetes distribution
print(df['Diagnosis'].value_counts())

##diabetes counts 1 and 0 in boxplot
class_counts = df['Diagnosis'].value_counts()
class_percentages = df['Diagnosis'].value_counts(normalize=True) * 100
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='Blues')
for index, value in enumerate(class_counts.values[::]):
    percentage = class_percentages[index]
    plt.text(index, value, f'{percentage:.2f}%', color='black', ha="center")
plt.xlabel('Diabetes status, 1="Yes", 0="No"')
plt.ylabel('Count')
plt.show()

##### Encode only the 'Gender' column
label_encoders = {}
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
label_encoders['Gender'] = le

#### Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
#plt.title('Feature Correlation Heatmap')
plt.show()

# Distribution of numeric features
numeric_columns = df.columns.drop(['Gender', 'High_BP', 'Smoking', 'Diagnosis'])
for column in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=column, kde=True, hue='Diagnosis', element='step')
    # plt.title(f'Distribution of {column} by Diabetes')
    plt.show()

def plot_feature_vs_diagnosis(df, features):
    for feature in features:
        plt.figure(figsize=(8, 5), dpi=100)
        sns.countplot(data=df, x=feature, hue='Diagnosis', palette='Set2')
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.legend(labels=["No Diabetes", "Diabetes"])
        plt.show()
# Usage:
plot_feature_vs_diagnosis(df, ['High_BP', 'Smoking', 'Gender'])
def calculate_percentages(df, feature):
    total_diabetes = df[df['Diagnosis'] == 1].shape[0]
    total_no_diabetes = df[df['Diagnosis'] == 0].shape[0]
    diabetes_with_feature = df[(df['Diagnosis'] == 1) & (df[feature] == 1)].shape[0]
    diabetes_without_feature = df[(df['Diagnosis'] == 1) & (df[feature] == 0)].shape[0]
    no_diabetes_with_feature = df[(df['Diagnosis'] == 0) & (df[feature] == 1)].shape[0]
    no_diabetes_without_feature = df[(df['Diagnosis'] == 0) & (df[feature] == 0)].shape[0]

    print(f"\n### Diabetes vs. {feature} ###")
    print(
        f"The percentage of having diabetes with {feature}: {np.round(100 * (diabetes_with_feature / total_diabetes), 2)}%")
    print(
        f"The percentage of having diabetes without {feature}: {np.round(100 * (diabetes_without_feature / total_diabetes), 2)}%")
    print("--------------------------------------------------")
    print(
        f"The percentage of having no diabetes with {feature}: {np.round(100 * (no_diabetes_with_feature / total_no_diabetes), 2)}%")
    print(
        f"The percentage of having no diabetes without {feature}: {np.round(100 * (no_diabetes_without_feature / total_no_diabetes), 2)}%")
# Usage:
calculate_percentages(df, 'High_BP')  # For High_BP
calculate_percentages(df, 'Smoking')  # For Smoking

### age and blood_glucose_level plot
sns.lineplot(x='Age', y= 'FBS', data=df)
plt.show()

# Boxplot to check for outliers
for column in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='Diagnosis', y=column, data=df)
    # plt.title(f'Boxplot of {column} by Diabetes')
    plt.xlabel("Diabetes")
    plt.ylabel(column)
    plt.show()

####  Normalization: Scale numeric features
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
print(df.head())

#### Split into training, validation, and test sets
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Step 5: Run a basic machine learning algorithm (Decision Tree)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Evaluate Decision Tree on test data
y_pred_dt = dt_model.predict(X_test)

#### Evaluate results with relevant metrics
def evaluate_model(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC AUC:", roc_auc_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

evaluate_model(y_test, y_pred_dt)
cm = confusion_matrix(y_test, y_pred_dt)
# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

### Build a neural network model
nn_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the neural network
history = nn_model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=32)

# Evaluate the neural network on test data
nn_eval = nn_model.evaluate(X_test, y_test)
print("Neural Network Test Accuracy:", nn_eval[1])

##Compare results with decision tree
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype('int32')
print("\nNeural Network Evaluation:")
evaluate_model(y_test, y_pred_nn)

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred_nn)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Create a single figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# Plot accuracy on the first subplot (ax1)
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
#ax1.set_title('Training and Validation Accuracy')
ax1.legend()
# Plot loss on the second subplot (ax2)
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
#ax2.set_title('Training and Validation Loss')
ax2.legend()
# Adjust spacing between subplots
plt.tight_layout()
# Show the combined plot
plt.show()

## Hyperparameter tuning
#### Hyperparameter 1 - neural number in hidden layer
# Store results for visualization
results = {'units': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []}

# Create a figure with 3 subplots in a row
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
unit_values = [128, 256, 1024]
for i, units in enumerate(unit_values):
    print(f"\nTesting with {units} units in first hidden layer")
    model = keras.Sequential([
        layers.Dense(units, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model and capture history
    history_units = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=32, verbose=0)
    y_pred_hp = (model.predict(X_test) > 0.5).astype('int32')
    # Evaluate the model
    acc = accuracy_score(y_test, y_pred_hp)
    prec = precision_score(y_test, y_pred_hp)
    rec = recall_score(y_test, y_pred_hp)
    f1 = f1_score(y_test, y_pred_hp)
    roc = roc_auc_score(y_test, y_pred_hp)
    print(
        f"Units: {units} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | "
        f"F1 Score: {f1:.4f} | ROC AUC: {roc:.4f}"
    )

    # Store results
    results['units'].append(units)
    results['accuracy'].append(acc)
    results['precision'].append(prec)
    results['recall'].append(rec)
    results['f1_score'].append(f1)
    results['roc_auc'].append(roc)

    # Plot training and validation loss in separate subplots
    axes[i].plot(history_units.history['loss'], label='Train Loss')
    axes[i].plot(history_units.history['val_loss'], label='Validation Loss')
    axes[i].set_xlabel('Epochs')
    axes[i].set_ylabel('Loss')
    axes[i].legend()
    axes[i].set_title(f'Units: {units}')
plt.tight_layout()
plt.show()

######### Hyperparameter 2 - learning rate
# Store results for visualization
results = {'learning_rate': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []}

# Create a figure with 3 subplots in a row
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)

# Learning rate values to test
learning_rates = [0.0001, 0.005, 0.1]

for i, learning_rate in enumerate(learning_rates):
    print(f"\nTesting with learning rate: {learning_rate}")

    # Define optimizer with custom learning rate
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Define model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Capture training history
    history_LR = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=32, verbose=0)

    y_pred_hp = (model.predict(X_test) > 0.5).astype('int32')

    # Evaluate the model
    acc = accuracy_score(y_test, y_pred_hp)
    prec = precision_score(y_test, y_pred_hp)
    rec = recall_score(y_test, y_pred_hp)
    f1 = f1_score(y_test, y_pred_hp)
    roc = roc_auc_score(y_test, y_pred_hp)

    print(
        f"Learning Rate: {learning_rate} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | "
        f"F1 Score: {f1:.4f} | ROC AUC: {roc:.4f}"
    )

    # Store results
    results['learning_rate'].append(learning_rate)
    results['accuracy'].append(acc)
    results['precision'].append(prec)
    results['recall'].append(rec)
    results['f1_score'].append(f1)
    results['roc_auc'].append(roc)

    # Plot training and validation loss in separate subplots
    axes[i].plot(history_LR.history['loss'], label='Train Loss')
    axes[i].plot(history_LR.history['val_loss'], label='Validation Loss')
    axes[i].set_xlabel('Epochs')
    axes[i].set_ylabel('Loss')
    axes[i].legend()
    axes[i].set_title(f'LR={learning_rate}')

plt.tight_layout()
plt.show()


############### hyperparameter 3 batch size
# Store results for visualization
results = {'batch_size': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []}

# Create a figure with 3 subplots in a row
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)

# Batch sizes to test
batch_sizes = [10, 100, 500]

for i, batch_size in enumerate(batch_sizes):
    print(f"\nTesting with batch size: {batch_size}")

    # Define model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Use the default learning rate of Adam
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model and store history
    history_new = model.fit(X_train, y_train, epochs=40, validation_data=(X_val, y_val), batch_size=batch_size, verbose=0)

    y_pred_hp = (model.predict(X_test) > 0.5).astype('int32')

    # Evaluate the model
    acc = accuracy_score(y_test, y_pred_hp)
    prec = precision_score(y_test, y_pred_hp)
    rec = recall_score(y_test, y_pred_hp)
    f1 = f1_score(y_test, y_pred_hp)
    roc = roc_auc_score(y_test, y_pred_hp)

    print(
        f"Batch Size: {batch_size} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | "
        f"F1 Score: {f1:.4f} | ROC AUC: {roc:.4f}"
    )
    # Store results
    results['batch_size'].append(batch_size)
    results['accuracy'].append(acc)
    results['precision'].append(prec)
    results['recall'].append(rec)
    results['f1_score'].append(f1)
    results['roc_auc'].append(roc)
    # Plot training and validation loss in separate subplots
    axes[i].plot(history_new.history['loss'], label='Train Loss')
    axes[i].plot(history_new.history['val_loss'], label='Validation Loss')
    axes[i].set_xlabel('Epochs')
    axes[i].set_ylabel('Loss')
    axes[i].legend()
    axes[i].set_title(f'Batch Size = {batch_size}')
plt.tight_layout()
plt.show()

# Dataset modification (delete records) - improve evaluation metrics
## find the normalized value for the original values i want to drop
mean_hba1c = scaler.mean_[numeric_columns.get_loc('HbA1c_level')]
std_hba1c = scaler.scale_[numeric_columns.get_loc('HbA1c_level')]
# Normalize the value 6
normalized_value_hba = (6 - mean_hba1c) / std_hba1c
mean_fbs = scaler.mean_[numeric_columns.get_loc('FBS')]
std_fbs = scaler.scale_[numeric_columns.get_loc('FBS')]
# Normalize the value 150
normalized_value_fbs = (150 - mean_fbs) / std_fbs

# Remove rows
df_cleaned = df[~((df['HbA1c_level'] > normalized_value_hba) & (df['Diagnosis'] == 0))]
df_cleaned = df_cleaned[~((df_cleaned['FBS'] > normalized_value_fbs) & (df_cleaned['Diagnosis'] == 0))]

# Split into features (X) and target (y)
X_cleaned = df_cleaned.drop('Diagnosis', axis=1)  # Assuming 'Diagnosis' is the target column
y_cleaned = df_cleaned['Diagnosis']

# Split the data into training, validation, and test sets
X_train_cleaned, X_temp_cleaned, y_train_cleaned, y_temp_cleaned = train_test_split(X_cleaned, y_cleaned, test_size=0.3, random_state=42, stratify=y_cleaned)
X_val_imp, X_test_imp, y_val_imp, y_test_imp = train_test_split(X_temp_cleaned, y_temp_cleaned, test_size=0.5, random_state=42, stratify=y_temp_cleaned)

# Rebuild and train the neural network with the cleaned dataset
nn_model_cleaned = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_cleaned.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


nn_model_cleaned.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history_cleaned = nn_model_cleaned.fit(X_train_cleaned, y_train_cleaned,
                                       epochs=30, validation_data=(X_val_imp, y_val_imp),
                                       batch_size=32)
# Evaluate the model with the cleaned data
nn_eval_cleaned = nn_model_cleaned.evaluate(X_test_imp, y_test_imp)
print("Test Accuracy After Cleaning:", nn_eval_cleaned[1])

# Predict using the trained model
y_pred_nn_cleaned = (nn_model_cleaned.predict(X_test_imp) > 0.5).astype('int32')

# Evaluate the model using custom metrics
print("\nNeural Network Evaluation:")
evaluate_model(y_test_imp, y_pred_nn_cleaned)

# Dataset modification (delete records) - worse evaluation metrics
## find the normalized value for the original values i want to drop
mean_age = scaler.mean_[numeric_columns.get_loc('Age')]
std_age = scaler.scale_[numeric_columns.get_loc('Age')]
normalized_value_age = (50 - mean_age) / std_age

mean_bmi = scaler.mean_[numeric_columns.get_loc('BMI')]
std_bmi = scaler.scale_[numeric_columns.get_loc('BMI')]
normalized_value_bmi = (40 - mean_bmi) / std_bmi

normalized_value_hba_2 = (7 - mean_hba1c) / std_hba1c
normalized_value_hba_3 = (5 - mean_hba1c) / std_hba1c

normalized_value_fbs_2 = (180 - mean_fbs) / std_fbs

df_cleaned = df[~((df['HbA1c_level'] > normalized_value_hba_2) & (df['Diagnosis'] == 1))]
df_cleaned = df_cleaned[~((df_cleaned['FBS'] > normalized_value_fbs_2) & (df_cleaned['Diagnosis'] == 1))]
df_cleaned = df_cleaned[~((df_cleaned['Age'] < normalized_value_age) & (df_cleaned['Diagnosis'] == 0))]
df_cleaned = df_cleaned[~((df_cleaned['Age'] < normalized_value_age) & (df_cleaned['Diagnosis'] == 1))]
df_cleaned = df_cleaned[~((df_cleaned['HbA1c_level'] < normalized_value_hba_3) & (df_cleaned['Diagnosis'] == 0))]
df_cleaned = df_cleaned[~((df_cleaned['BMI'] < normalized_value_bmi) & (df_cleaned['Diagnosis'] == 1))]
df_cleaned = df_cleaned[~((df_cleaned['BMI'] < normalized_value_bmi) & (df_cleaned['Diagnosis'] == 0))]

# Split into features (X) and target (y)
X_cleaned = df_cleaned.drop('Diagnosis', axis=1)  # Assuming 'Diagnosis' is the target column
y_cleaned = df_cleaned['Diagnosis']

# Split the data into training, validation, and test sets
X_train_cleaned, X_temp_wors, y_train_cleaned, y_temp_wors = train_test_split(X_cleaned, y_cleaned, test_size=0.3, random_state=42, stratify=y_cleaned)
X_val_wors, X_test_wors, y_val_wors, y_test_wors = train_test_split(X_temp_wors, y_temp_wors, test_size=0.5, random_state=42, stratify=y_temp_wors)

# Rebuild and train the neural network with the cleaned dataset
nn_model_cleaned = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_cleaned.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

nn_model_cleaned.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history_cleaned = nn_model_cleaned.fit(X_train_cleaned, y_train_cleaned,
                                       epochs=30, validation_data=(X_val_wors, y_val_wors),
                                       batch_size=32)
# Evaluate the model with the cleaned data
nn_eval_cleaned = nn_model_cleaned.evaluate(X_test_wors, y_test_wors)
print("Test Accuracy After Cleaning:", nn_eval_cleaned[1])

# Predict using the trained model
y_pred_nn_cleaned = (nn_model_cleaned.predict(X_test_wors) > 0.5).astype('int32')

# Evaluate the model using custom metrics
print("\nNeural Network Evaluation:")
evaluate_model(y_test_wors, y_pred_nn_cleaned)

##### Improve network architecture
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Build a neural network model
nn_model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_smote.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the neural network
history_arch = nn_model.fit(X_train_smote, y_train_smote, epochs=40, validation_data=(X_val, y_val), batch_size=500)
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype('int32')
print("\nNeural Network Evaluation:")
evaluate_model(y_test, y_pred_nn)

# Create a single plot for loss
plt.figure(figsize=(8, 5))  # Adjust figure size as needed
plt.plot(history_arch.history['loss'], label='Train Loss')
plt.plot(history_arch.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss architecture improve')
plt.show()

#### Suggest a new metric
# Define the custom callback to calculate 2-F1 Learning Efficiency Ratio (LER-2F1)
class LearningEfficiency2F1Callback(Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.f1_scores = []
        self.ler_2f1_values = []

    def on_epoch_end(self, epoch, logs=None):
        # Get predictions for the validation set and compute F1 score
        y_pred = (self.model.predict(self.X_val) > 0.5).astype('int32')
        f1 = f1_score(self.y_val, y_pred)
        self.f1_scores.append(f1)

        # Calculate the LER-2F1 if it's not the first epoch
        if epoch > 0:
            f1_change = self.f1_scores[-1] - self.f1_scores[-2]
            ler_2f1 = f1_change / (epoch + 1)
            self.ler_2f1_values.append(ler_2f1)
        else:
            self.ler_2f1_values.append(0)  # No change in F1 score for the first epoch

# Define and compile the neural network model
nn_model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Instantiate the custom callback with validation data
ler_2f1_callback = LearningEfficiency2F1Callback(X_val, y_val)

# Train the model with the custom 2-F1 LER callback
history_2f1 = nn_model.fit(
    X_train, y_train, epochs=30, validation_data=(X_val, y_val),
    batch_size=32, callbacks=[ler_2f1_callback]
)

# Plot 2-F1 LER-2F1 (Learning Efficiency Ratio) over epochs
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ler_2f1_callback.ler_2f1_values) + 1), ler_2f1_callback.ler_2f1_values, label='2-F1 Learning Efficiency Ratio (LER-2F1)')
plt.xlabel('Epochs')
plt.ylabel('LER-2F1')
#plt.title('2-F1 Learning Efficiency Ratio Over Epochs')
plt.legend()
plt.show()

##### Change data balance
smote = SMOTE(sampling_strategy=1, random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

rus = RandomUnderSampler(sampling_strategy=3/7, random_state=42)
X_train_mod, y_train_mod = rus.fit_resample(X_train, y_train)

rus2 = RandomUnderSampler(sampling_strategy=1/9, random_state=42)
X_train_sev, y_train_sev = rus2.fit_resample(X_train, y_train)

def train_model(X_train, y_train, title):
    nn_model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = nn_model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=32, verbose=0)

    y_pred = (nn_model.predict(X_test) > 0.5).astype('int32')

    print(f"\nResults for {title}:")
    print(classification_report(y_test, y_pred))
    return history

# Train models on different imbalanced datasets
hist_bal = train_model(X_train_bal, y_train_bal, "Balanced Data (50:50)")
hist_mod = train_model(X_train_mod, y_train_mod, "Moderate Imbalance (70:30)")
hist_sev = train_model(X_train_sev, y_train_sev, "Severe Imbalance (90:10)")

plt.figure(figsize=(8,5))
plt.plot(hist_bal.history['val_accuracy'], label="Balanced (50:50)")
plt.plot(hist_mod.history['val_accuracy'], label="Moderate Imbalance (70:30)")
plt.plot(hist_sev.history['val_accuracy'], label="Severe Imbalance (90:10)")
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
#plt.title('Model Performance with Different Class Imbalances')
plt.show()

######### Dimensionality reduction
# Choose the number of principal components (e.g., 2 for visualization)
pca = PCA(n_components=2)

# Fit and transform the training data
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

# Convert to DataFrame for visualization
df_pca = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2'])
df_pca['Diagnosis'] = y_train.values  # Add target labels for plotting
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

plt.figure(figsize=(8,6))
sns.scatterplot(x=df_pca['PC1'], y=df_pca['PC2'], hue=df_pca['Diagnosis'], palette='coolwarm', alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of the Dataset")
plt.legend(title="Diagnosis")
plt.show()

nn_model_pca = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train_pca.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
nn_model_pca.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train model with PCA data
history_pca = nn_model_pca.fit(X_train_pca, y_train, epochs=30, validation_data=(X_val_pca, y_val), batch_size=32, verbose=0)
# Evaluate model
y_pred_pca = (nn_model_pca.predict(X_test_pca) > 0.5).astype('int32')
print("\nClassification Report (PCA):")
print(classification_report(y_test, y_pred_pca))
