import pandas as pd

df = pd.read_csv('churnguard_data.csv')

# 1. Drop customerID
df.drop(columns=['customerID'], inplace=True)

# 2. Remove duplicates
df.drop_duplicates(inplace=True)

# 3. Strip whitespace
df['gender'] = df['gender'].str.strip()
df['PaymentMethod'] = df['PaymentMethod'].str.strip()

# 4. Standardise casing
df['Churn'] = df['Churn'].str.strip().str.title()
df['PhoneService'] = df['PhoneService'].str.strip().str.title()
df['PaperlessBilling'] = df['PaperlessBilling'].str.strip().str.title()

# 5. Fix Contract
def fix_contract(val):
    val = str(val).strip().lower()
    if val in ['month-to-month', 'month to month', 'monthly', 'mtm']:
        return 'Month-to-month'
    elif val in ['one year', '1 year', 'one-year']:
        return 'One year'
    elif val in ['two year', '2 year', 'two-year']:
        return 'Two year'
    return val

df['Contract'] = df['Contract'].apply(fix_contract)

# 6. Fix InternetService
def fix_internet(val):
    val = str(val).strip().lower()
    if val in ['dsl']:
        return 'DSL'
    elif val in ['fibre optic', 'fiberoptic', 'fiber optic', 'fibreoptic']:
        return 'Fiber optic'
    elif val in ['no', 'none']:
        return 'No'
    return val

df['InternetService'] = df['InternetService'].apply(fix_internet)

# 7. Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 8. Remove bad tenure rows
df = df[df['tenure'] > 0]

# 9. Remove outlier MonthlyCharges
df = df[(df['MonthlyCharges'] >= 10) & (df['MonthlyCharges'] <= 200)]

# 10. Fill missing values
df['MonthlyCharges'].fillna(df['MonthlyCharges'].mean(), inplace=True)
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
df['tenure'].fillna(round(df['tenure'].median()), inplace=True)

print(df.shape)
print(df.isnull().sum())