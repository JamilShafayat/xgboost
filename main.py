import pandas as pd
import os
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import xgboost
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import pickle
from flask import Flask
from flask import request

app = Flask(__name__)
print(xgboost.__version__)

def model_train():
  # train = pd.read_csv("creditcard.csv")
  train = pd.read_csv("fraudTest.csv")
  print(train)
  label_encoder = LabelEncoder()
  scaler= StandardScaler()
  df = train.apply(lambda x: label_encoder.fit_transform(x) if x.dtype == 'O' else x)
  trainup = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
  cov_matrix = abs(trainup.cov())
  # plt.figure(figsize=(20, 20))
  sns.heatmap(cov_matrix, annot=True, cmap='Blues', fmt='.2f', linewidths=.5)
  # plt.title('Covariance Matrix Heatmap')
  # plt.savefig("Covariance Matrix Heatmap")
  # X=df[['V10', 'V12', 'V14', 'V16', 'V17']]
  X=df[['cc_num', 'amt', 'unix_time', 'merch_lat', 'merch_long']]
  Y=df['is_fraud']
  print(df.info())
  X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

  model = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, min_child_weight=3,subsample=0.8,colsample_bytree=0.9,n_estimators=500)
  kf = KFold(n_splits=5, shuffle=True, random_state=42)
  cv_results = cross_val_score(model, X, Y, cv=kf, scoring='accuracy')
  accuracy_values = cv_results
  average_accuracy = cv_results.mean()
  print("Accuracy for each fold:", accuracy_values)
  print("Average Accuracy across all folds:", average_accuracy)
  X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)
  model.fit(X_train, y_train)
  prediction = model.predict(X_test)
  # cm= metrics.confusion_matrix(y_test, prediction)
  # cm_dis = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=df['Class'].unique())
  # cm_dis.plot()
  # plt.title('Predicted Table')
  # plt.savefig("Predicted Table")
  classx= metrics.classification_report(y_test, prediction)
  print('Classification Report:\n', classx)
  pickle.dump(model, open("models/001.pkl", "wb"))

def prediction(X):
  try:
    model = pickle.load(open("models/001.pkl", "rb"))
    print(model)
    return model.predict(X)
  except Exception as e:
    print(e)
  return -1

def main():
  if not os.path.exists("models/001.pkl"):
    model_train()
  app.run(host="0.0.0.0", port=9021, debug=True)

def string_to_float_list(data_string):
  return [float(num) for num in data_string.split(",")]

@app.route("/")
def home():
    x_str = request.args.get('x')
    print(x_str)
    x = [string_to_float_list(x_str)]
    print(x)
    # y = 21
    y = prediction(x)
    # # print(y)
    return f"{x_str} -> {x} => {y}" 

if __name__ == '__main__':
  main()
