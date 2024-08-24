# 분류 알고리즘


## 목표
- Random Forest | Decision Tree | Logistic Regression 알고리즘 별 분류 정확도 확인

## 데이터셋
- Source: [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- Description
  - 변수
    - 클래스 개수: 2개 [red : 0, white : 1]
    - 타켓 변수 : type: 종류
      
  - 설명변수(Feature) 13개
    - fixed acidity: 산도
    - volatile acidity: 휘발성산
    - citric acid: 시트르산
    - residual sugar: 잔당(발효 후 남은 당분)
    - chlorides: 염화물
    - free sulfur dioxide: 독립 이산화황
    - total sulfur dioxide: 총 이산화황
    - density: 밀도
    - pH: 수소이온농도
    - sulphates: 황산염
    - alcohol: 알코올
    - quality: 품질


## 모델 성능
- Random Forest
  - Accuracy: 97.802%
  - Precision: 100%
  - Recall: 96.43%
  - Confusion Matrix
  - ![Figure_1](https://github.com/user-attachments/assets/ca2c2293-1f56-4783-a873-dcfec814a603)

- Decision Tree
  - Accuracy: 97.802%
  - Precision: 97.56%
  - Recall: 96.43%
  - Confusion Matrix
  - ![Figure_2](https://github.com/user-attachments/assets/bb8f72d4-177e-4463-add7-4b938a9c9869)
 
- Logistic Regression
  - Accuracy: 97.802%
  - Precision: 100%
  - Recall: 96.43%
  - Confusion Matrix
  - ![Figure_3](https://github.com/user-attachments/assets/18261a0b-8a53-4c37-9dcc-d570b1b63a12)

  
