import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from LogisticRegression import LogisticRegression

def main():
    data = pd.read_csv("data.csv")

    data.diagnosis = [1 if value == "M" else 0 for value in data.diagnosis]
    data["diagnosis"] = data["diagnosis"].astype("category", copy=False)

    y = data["diagnosis"]
    x = data.drop(["diagnosis"], axis=1)
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

    log_model = LogisticRegression(lr=0.01)

    log_model.fit(x_train, y_train.values.astype(float))

    predictions = log_model.predict(x_test)

    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))


main()