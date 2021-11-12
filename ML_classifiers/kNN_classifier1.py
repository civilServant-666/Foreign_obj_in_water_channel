import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import time
start_time = time.time()

# Read and exam the data
fp_train = "./classifier1_iswater_training_lbp.csv";
fp_test = "./classifier1_iswater_test_lbp_original_expand.csv";
df_train = pd.read_csv(fp_train, header=0, encoding="gbk");
df_test = pd.read_csv(fp_test, header=0, encoding="gbk");

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score

X_train = df_train.drop("class", axis=1);
y_train = df_train["class"];
X_test = df_test.drop("class", axis=1);
y_test = df_test["class"];

### NEW - Train with cross-validation
knnclassifier = KNeighborsClassifier(n_neighbors=4)
scoring = ['accuracy', 'f1_weighted', 'f1_macro', 'f1']
k_val = 5
scores = cross_validate(knnclassifier,X_train,y_train,cv=k_val,scoring=scoring)
sorted(scores.keys())
for i in range(k_val):
    print('{},fit_time:{},score_time:{},acc:{},f1_wei:{},f1_mac:{},f1:{}'\
          .format(i,scores['fit_time'][i],scores['score_time'][i]\
          ,scores['test_accuracy'][i]\
          ,scores['test_f1_weighted'][i]\
          ,scores['test_f1_macro'][i]\
          ,scores['test_f1'][i]))
print('ave_fit_time:{}'.format(scores['fit_time'].mean()))
print('ave_acc:{}'.format(scores['test_accuracy'].mean()))
print('ave_f1_wei:{}'.format(scores['test_f1_weighted'].mean()))
print('ave_f1_mac:{}'.format(scores['test_f1_macro'].mean()))
print('ave_f1:{}'.format(scores['test_f1'].mean()))
# Train with complete dataset and predict testset
knnclassifier.fit(X_train,y_train)
y_pred = knnclassifier.predict(X_test)
# Evaluate the results
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
# Predict instance class and output to exl file
'''pf = pd.DataFrame(y_pred,index=range(3390),columns=['predict_class']);
pf.to_excel('./classifier1_iswater_testset_predict_sep.xls');'''

