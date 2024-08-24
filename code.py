import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- Random Forest -----------------------------------

# 데이터 불러오기
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 피쳐, 타켓 데이터 분리
train_input = train.drop(["type", "Index"], axis=1)
train_target = train['type']
test_df = test.drop(["Unnamed: 13", "Index"], axis=1)

# 카테고리 데이터 -> 수치형 데이터 변환 [red : 0], [white : 1]
from sklearn.preprocessing import LabelEncoder
label_index = LabelEncoder()
train_target = label_index.fit_transform(train_target)

# 데이터 셔플(데스트 샘플 0.2 비율)
from sklearn.model_selection import train_test_split
X_in, Y_in, X_val, Y_val = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42
)

# Random Forest 
from sklearn.ensemble import RandomForestClassifier
fc = RandomForestClassifier(random_state=42)
fc.fit(X_in, X_val)

# 예측 코스어 확인
from sklearn.metrics import accuracy_score, f1_score, recall_score
pred_test_1 = fc.predict(Y_in)
accuracy_1 = accuracy_score(Y_val, pred_test_1)
recall_1 = recall_score(Y_val, pred_test_1)

# 결과 dataframe
result_1 = fc.predict(test_df)

# ---------------------Decision Tree--------------------------------

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_in, X_val)

# 예측 스코어 
pred_test_2 = fc.predict(Y_in)
accuracy_2  = accuracy_score(Y_val, pred_test_2)
recall_2 = recall_score(Y_val, pred_test_2)

# 결과 dataframe
result_2 = tree.predict(test_df)

# ---------------------Logistic Regression--------------------------------

# 스케일 표준화
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
st.fit(X_in)
x_scaled = st.transform(X_in)
y_scaled = st.transform(Y_in)
test_df_scaled = st.transform(test_df)

# Logistic Regreesion
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(random_state=42)
lg.fit(x_scaled, X_val)

# 예측 스코어
pred_test_3 = lg.predict(y_scaled)
accuracy_3 = accuracy_score(Y_val, pred_test_3)
recall_3 = recall_score(Y_val, pred_test_3)

# 결과 dataframe
result_3 = lg.predict(test_df_scaled)

# -------------------- Result ------------------------

# 훈련 스코어
#print(fc.score(X_in, X_val), tree.score(X_in, X_val), lg.score(x_scaled, X_val))

# 테스트 스코어
#print(fc.score(Y_in, Y_val), tree.score(Y_in, Y_val), lg.score(y_scaled, Y_val))

# 시긱화
from sklearn.metrics import confusion_matrix

plt.figure(figsize=(4,4))
cm = confusion_matrix(result_1, result_3)
group_names = ["TN", "FP", "FN", "TP"]
group_counts = [value for value in cm.flatten()]
group_percentages = [f"{value:.1%}" for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='PuBuGn')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()





# -------------------- save file ------------------------

save_df = pd.DataFrame(
    {"Index" : test["Index"],
     "Random Forest" : result_1,
     "Decision Tree" : result_2,
     "Logistic" : result_3}
) 

save_df.to_csv(path_or_buf='save_df.csv',index=False)



