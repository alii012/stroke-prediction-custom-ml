import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


data = pd.read_csv(
    'healthcare-dataset-stroke-data.csv', parse_dates=True)
data = data.drop(columns=['id'])

data['bmi'] = data['bmi'].fillna(data['bmi'].median())
stroke = data['stroke'].value_counts()
data = data[data['gender'] != 'Other']

# covnert them into 0s and 1s
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
data['ever_married'] = data['ever_married'].map({'Yes': 1, "No": 0})
data['Residence_type'] = data['Residence_type'].map({'Urban': 1, 'Rural': 0})

# get dummies for work_type and somking statues means each vlaue of the colunm will be its column using 0 1
data = pd.get_dummies(data, columns=['work_type'], drop_first=True)
data = pd.get_dummies(data, columns=['smoking_status'], drop_first=True)

# split the data X, y and get X_train, X_test
X = data.drop(columns='stroke')
y = data['stroke']


X['age_glucose'] = X['age'] * X['avg_glucose_level']
X['age_s'] = X['age'] ** 2

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# scale the features using z-score formula (X - mean)/ std
feature_to_scale = ['age', 'avg_glucose_level', "bmi", "age_glucose", 'age_s']

mean = np.mean(X_train[feature_to_scale], axis=0)
std = np.std(X_train[feature_to_scale], axis=0)

X_train[feature_to_scale] = (X_train[feature_to_scale] - mean) / std
X_test[feature_to_scale] = (X_test[feature_to_scale] - mean) / std


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def prediction(X, w, b):
    f_x = X @ w + b
    return sigmoid(f_x)


def cost(X, y, w, b):
    m = X.shape[0]

    f_x = prediction(X, w, b)

    total_cost = -1/m * np.sum(y * np.log(f_x) + (1 - y) * np.log(1 - f_x))
    return total_cost


def gradient(X, y, w, b, alpha, num_itrate):
    m = X.shape[0]

    for i in range(num_itrate):
        f_x = prediction(X, w, b)
        err = f_x - y
        dw = 1 / m * X.T @ err
        db = 1 / m * np.sum(err)

        w = w - alpha * dw
        b = b - alpha * db

    return w, b


def predict(X, w, b):
    probs = prediction(X, w, b)
    return (probs >= 0.3).astype(int)


# initialize

w = np.zeros(X_train.shape[1])
b = 0
alpha = 0.01
iteration = 10000

w, b = gradient(X_train.values, y_train.values, w, b, alpha, iteration)

y_train_pred = predict(X_train.values, w, b)
y_test_pred = predict(X_test.values, w, b)

print(np.mean(y_train_pred == y_train.values))
print(np.mean(y_test_pred == y_test.values))

print("Matrix", confusion_matrix(y_test.values, y_test_pred))
print("Precivsion", precision_score(y_test.values, y_test_pred))
print("Recall", recall_score(y_test.values, y_test_pred))
feature_names = X_train.columns
top_features = sorted(zip(feature_names, w), key=lambda x: abs(x[1]), reverse=True)
print("\nTop 3 features:")
for name, weight in top_features[:3]:
    print(f"{name}: {weight:.4f}")

# trying to fix my model by adding more weights and adding more traning exmaples where its 50 50

def weight_gradient(X, y, w, b, alpha, num_itrate, weight_vector):
    m = X.shape[0]

    for i in range(num_itrate):
        f_x = prediction(X, w, b)
        err = (f_x - y) * weight_vector
        dw = 1 / m * X.T @ err
        db = 1 / m * np.sum(err)

        w = w - alpha * dw
        b = b - alpha * db

    return w, b


def weight_cost(X, y, w, b, weight_vector):
    m = X.shape[0]

    f_x = prediction(X, w, b)

    total_cost = -1 / m * np.sum(weight_vector * (y * np.log(f_x) + (1 - y) * np.log(1 - f_x)))
    return total_cost

# create additional features

weights = np.where(y_train == 1, 8, 1)
w = np.zeros(X.shape[1])
b = 0

w,b = weight_gradient(X_train.values, y_train.values, w, b, alpha, iteration, weights)

y_train_pred = predict(X_train.values, w, b)
y_test_pred = predict(X_test.values, w, b)

print(np.mean(y_train_pred == y_train.values))
print(np.mean(y_test_pred == y_test.values))

print("Matrix", confusion_matrix(y_test.values, y_test_pred))
print("Precivsion", precision_score(y_test.values, y_test_pred))
print("Recall", recall_score(y_test.values, y_test_pred))

# 1. Visualization: Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test.values, y_test_pred), annot=True, fmt='d', xticklabels=['Healthy', 'Stroke'],
            yticklabels=['Healthy', 'Stroke'], cmap='Blues')
plt.title(f'Final Model Confusion Matrix\n(Recall: {recall_score(y_test, y_test_pred):.2f})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# 2. Visualization: Feature Importance
# We use the absolute value of weights to show "Impact" regardless of direction
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'Weight': np.abs(w)

}).sort_values(by='Weight', ascending='False')

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Weight', y='feature', palette='viridis')
plt.title('Which Features Matter Most? (Absolute Weights)')
plt.xlabel('Impact (Absolute Weight Value)')
plt.show()


model_params = {
    'weights': w,
    'bias': b,
    'mean': mean.values,
    'std': std.values,
    'feature_names': X_train.columns.tolist()
}

np.savez('stroke_model_v1.npz', **model_params)
print("Model saved successfully as 'stroke_model_v1.npz'!")