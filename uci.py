import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference

# Step 1: Load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
           "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
           "hours-per-week", "native-country", "income"]
df = pd.read_csv(url, names=columns, sep=r",\s*", engine="python")
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# Step 2: Encode categoricals
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Step 3: Split data
x = df.drop('income', axis=1)
y = df['income']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 4: Scale
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# Step 5: Train model
model = LogisticRegression(max_iter=1000)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

# Step 6: Evaluate accuracy
print("Accuracy:", accuracy_score(ytest, ypred))

# Step 7: Fairness audit (on gender)
index = x.columns.get_loc("sex")
sensitive_features = xtest[:, index]

frame = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
    y_true=ytest,
    y_pred=ypred,
    sensitive_features=sensitive_features
)

print("\n Fairness Metrics by Group:")
print(frame.by_group)

print("\n Demographic Parity Difference:",
      demographic_parity_difference(ytest, ypred, sensitive_features=sensitive_features))