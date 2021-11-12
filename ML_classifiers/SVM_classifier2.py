import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import time
start_time = time.time()

# Read and exam the data
fp_train = "./classifier2_type_training_lbp_instances_cascade.csv";
fp_test = "./classifier2_type_test_lbp_instances_cascade.csv";
df_train = pd.read_csv(fp_train, header=0, encoding="gbk");
df_test = pd.read_csv(fp_test, header=0, encoding="gbk");

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score

X_train = df_train.drop("class", axis=1);
y_train = df_train["class"];
X_test = df_test.drop("class", axis=1);
y_test = df_test["class"];

### NEW - Train with cross-validation
svclassifier = SVC(kernel='rbf',C=1000,gamma=17,class_weight='balanced')
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
k_val = 10
scores = cross_validate(svclassifier,X_train,y_train,cv=k_val,scoring=scoring)
sorted(scores.keys())
for i in range(k_val):
    print('{},fit_time:{},score_time:{},acc:{},f1:{},pre:{},rec:{}'\
          .format(i,scores['fit_time'][i],scores['score_time'][i]\
          ,scores['test_accuracy'][i]\
          ,scores['test_f1_weighted'][i]\
          ,scores['test_precision_weighted'][i]\
          ,scores['test_recall_weighted'][i]))
print('ave_fit_time:{}'.format(scores['fit_time'].mean()))
print('ave_acc:{}'.format(scores['test_accuracy'].mean()))
print('ave_f1:{}'.format(scores['test_f1_weighted'].mean()))
print('ave_pre:{}'.format(scores['test_precision_weighted'].mean()))
print('ave_rec:{}'.format(scores['test_recall_weighted'].mean()))
# Train with complete dataset and predict testset
svclassifier.fit(X_train,y_train)
y_pred = svclassifier.predict(X_test)
# Evaluate the results
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
# Predict instance class and output to exl file
'''y_instance = svclassifier.predict(X_instance)
pf = pd.DataFrame(y_instance,index=range(73),columns=['predict_class']);
pf.to_excel('./classifier2_type_testset_instances_predict_sep_add.xls');'''
