import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('churnguard_data.csv')

# Drop customerID
df.drop(columns=['customerID'], inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)

# Strip whitespace
df['gender'] = df['gender'].str.strip()
df['PaymentMethod'] = df['PaymentMethod'].str.strip()
df['Churn'] = df['Churn'].str.strip().str.title()
df['PhoneService'] = df['PhoneService'].str.strip().str.title()
df['PaperlessBilling'] = df['PaperlessBilling'].str.strip().str.title()

# Fix Contract
def fix_contract(val):
    val = str(val).strip().lower()
    if val in ['month-to-month', 'month to month', 'monthly', 'mtm']:
        return 'Month-to-month'
    elif val in ['one year', '1 year', 'one-year']:
        return 'One year'
    elif val in ['two year', '2 year', 'two-year']:
        return 'Two year'
    return 'Month-to-month'  # default fallback

df['Contract'] = df['Contract'].apply(fix_contract)

# Fix InternetService
def fix_internet(val):
    val = str(val).strip().lower()
    if val == 'dsl': return 'DSL'
    elif val in ['fibre optic', 'fiberoptic', 'fiber optic', 'fibreoptic']: return 'Fiber optic'
    elif val in ['no', 'none']: return 'No'
    return 'No'  # default fallback

df['InternetService'] = df['InternetService'].apply(fix_internet)

# Fix TotalCharges - convert to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Remove bad tenure rows
df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
df = df[df['tenure'] > 0]

# Remove MonthlyCharges outliers
df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
df = df[(df['MonthlyCharges'] >= 10) & (df['MonthlyCharges'] <= 200)]

# Fill missing values BEFORE encoding
df['MonthlyCharges'].fillna(df['MonthlyCharges'].mean(), inplace=True)
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
df['tenure'].fillna(int(round(df['tenure'].median())), inplace=True)

# Fill any remaining NaN in categorical columns
df['gender'].fillna('Male', inplace=True)
df['PhoneService'].fillna('No', inplace=True)
df['PaperlessBilling'].fillna('No', inplace=True)
df['PaymentMethod'].fillna(df['PaymentMethod'].mode()[0], inplace=True)

# Drop rows where Churn is NaN (can't train without target)
df.dropna(subset=['Churn'], inplace=True)

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categoricals
df = pd.get_dummies(df, columns=['gender', 'PhoneService', 'InternetService',
                                  'Contract', 'PaperlessBilling', 'PaymentMethod'],
                    drop_first=True)

# Final safety net - drop any remaining NaN rows
df.dropna(inplace=True)

print("Shape after cleaning:", df.shape)
print("Remaining NaNs:", df.isnull().sum().sum())

X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Stay', 'Churn']))