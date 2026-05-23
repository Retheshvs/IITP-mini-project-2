import pandas as pd
from sklearn.linear_model import LogisticRegression

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
    return 'Month-to-month'

df['Contract'] = df['Contract'].apply(fix_contract)

# Fix InternetService
def fix_internet(val):
    val = str(val).strip().lower()
    if val == 'dsl': return 'DSL'
    elif val in ['fibre optic', 'fiberoptic', 'fiber optic', 'fibreoptic']: return 'Fiber optic'
    elif val in ['no', 'none']: return 'No'
    return 'No'

df['InternetService'] = df['InternetService'].apply(fix_internet)

# Convert numeric columns safely
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce')

# Remove bad rows
df = df[df['tenure'] > 0]
df = df[(df['MonthlyCharges'] >= 10) & (df['MonthlyCharges'] <= 200)]

# Fill missing values
df['MonthlyCharges'].fillna(df['MonthlyCharges'].mean(), inplace=True)
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
df['tenure'].fillna(int(round(df['tenure'].median())), inplace=True)
df['SeniorCitizen'].fillna(0, inplace=True)

# Drop rows where Churn is NaN
df.dropna(subset=['Churn'], inplace=True)

# Encode Churn
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode Contract as ordinal
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
df['Contract'] = df['Contract'].map(contract_map)

# Use only the 5 features as required by Task 4
features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Contract']

df_model = df[features + ['Churn']].copy()

# Final safety net
df_model.dropna(inplace=True)

print("Shape:", df_model.shape)
print("NaNs remaining:", df_model.isnull().sum())

X = df_model[features]
y = df_model['Churn']

# Train on full dataset
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Collect user input
tenure = int(input("Enter tenure (months): "))
monthly = float(input("Enter Monthly Charges: "))
total = float(input("Enter Total Charges: "))
senior = int(input("Senior Citizen? (1 = Yes, 0 = No): "))
contract = int(input("Contract type (0 = Month-to-month, 1 = One year, 2 = Two year): "))

user_input = [[tenure, monthly, total, senior, contract]]
prediction = model.predict(user_input)[0]

if prediction == 1:
    print("Prediction: This customer is likely to CHURN.")
else:
    print("Prediction: This customer is likely to STAY.")