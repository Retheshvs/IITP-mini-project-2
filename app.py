from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app, origins="*")  # Allow all origins

def load_and_train():
    df = pd.read_csv('churnguard_data.csv')

    df.drop(columns=['customerID'], inplace=True)
    df.drop_duplicates(inplace=True)

    df['gender'] = df['gender'].str.strip()
    df['PaymentMethod'] = df['PaymentMethod'].str.strip()
    df['Churn'] = df['Churn'].str.strip().str.title()
    df['PhoneService'] = df['PhoneService'].str.strip().str.title()
    df['PaperlessBilling'] = df['PaperlessBilling'].str.strip().str.title()

    def fix_contract(val):
        val = str(val).strip().lower()
        if val in ['month-to-month', 'month to month', 'monthly', 'mtm']:
            return 'Month-to-month'
        elif val in ['one year', '1 year', 'one-year']:
            return 'One year'
        elif val in ['two year', '2 year', 'two-year']:
            return 'Two year'
        return 'Month-to-month'

    def fix_internet(val):
        val = str(val).strip().lower()
        if val == 'dsl': return 'DSL'
        elif val in ['fibre optic', 'fiberoptic', 'fiber optic']: return 'Fiber optic'
        return 'No'

    df['Contract'] = df['Contract'].apply(fix_contract)
    df['InternetService'] = df['InternetService'].apply(fix_internet)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce')

    df = df[df['tenure'] > 0]
    df = df[(df['MonthlyCharges'] >= 10) & (df['MonthlyCharges'] <= 200)]

    df['MonthlyCharges'].fillna(df['MonthlyCharges'].mean(), inplace=True)
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
    df['tenure'].fillna(int(round(df['tenure'].median())), inplace=True)
    df['SeniorCitizen'].fillna(0, inplace=True)

    df.dropna(subset=['Churn'], inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    df['Contract'] = df['Contract'].map(contract_map)

    features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Contract']
    df_model = df[features + ['Churn']].dropna()

    X = df_model[features]
    y = df_model['Churn']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    print("✅ Model trained successfully!")
    return model

model = load_and_train()

@app.route('/')
def home():
    return "ChurnGuard API is running!"

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    data = request.json
    print("Received data:", data)  # Debug log

    try:
        features = [[
            int(data['tenure']),
            float(data['monthlyCharges']),
            float(data['totalCharges']),
            int(data['seniorCitizen']),
            int(data['contract'])
        ]]
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        result = "CHURN" if prediction == 1 else "STAY"

        response = jsonify({
            "prediction": result,
            "probability": round(float(probability) * 100, 2)
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except Exception as e:
        print("Error:", str(e))  # Debug log
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)