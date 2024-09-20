## 은행 마케팅 데이터셋을 활용한 분류기 만들기

### 독립변수:

- age (숫자형): 나이
- job: 직업 유형 (범주형: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
- marital: 결혼 상태 (범주형: 'divorced','married','single','unknown'; 참고: 'divorced'는 이혼 또는 사별을 의미함)
- education: 교육 수준 (범주형: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
- default: 연체된 대출이 있는지 여부 (범주형: 'no','yes','unknown')
- housing: 주택 대출이 있는지 여부 (범주형: 'no','yes','unknown')
- loan: 개인 대출이 있는지 여부 (범주형: 'no','yes','unknown')
- contact: 연락 방식 (범주형: 'cellular','telephone')
- month: 마지막 연락 월 (범주형: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
- day_of_week: 마지막 연락 요일 (범주형: 'mon','tue','wed','thu','fri')
- duration: 마지막 연락 시간(초 단위, 숫자형).
    - 중요한 주의사항: 이 속성은 출력(target)에 큰 영향을 미침 (예: duration=0이면 y='no'). 그러나 통화가 이루어지기 전에는 duration을 알 수 없고, 통화가 끝나면 y값은 명확해짐. 따라서 이 속성은 벤치마크 목적으로만 포함되어야 하며, 현실적인 예측 모델을 만들 때는 제외해야 함.
- campaign: 이번 캠페인에서 해당 고객에게 이루어진 연락 횟수 (숫자형, 마지막 연락 포함)
- pdays: 이전 캠페인에서 고객에게 마지막으로 연락한 후 경과한 일수 (숫자형; 999는 이전에 연락하지 않은 경우를 의미)
- previous: 이번 캠페인 전에 해당 고객에게 이루어진 연락 횟수 (숫자형)
- poutcome: 이전 마케팅 캠페인의 결과 (범주형: 'failure','nonexistent','success')

### 종속변수(target, 예측 대상):

- y: 고객이 정기 예금을 신청했는지 여부 (이진형: 'yes','no')

---

```python
import pandas as pd  # 2.1.4 ver
import numpy as np  # 1.25.2 ver
import seaborn as sns  # 0.13.1 ver
import matplotlib.pyplot as plt  # v3.7.1 ver
import warnings
import pycaret  # 3.3.2 ver
from pycaret.classification import *

warnings.filterwarnings('ignore')
```

```python
df = pd.read_csv('bank.csv')
df.head()
```
![1](https://github.com/user-attachments/assets/5c513a24-63dc-4d6a-87f9-2fd1c6892edd)

```python
# 'duration' 컬럼 버리기(drop)
df.drop('duration', axis=1, inplace=True) # duration은 사후에 기록된 컬럼이기 때문에, 앞으로의 예측에서는 수집할 수 없는 데이터
df.info()
```

![image2](https://github.com/user-attachments/assets/6f633016-88ef-42ae-bf92-de01bd754aac)

![image3](https://github.com/user-attachments/assets/71e59feb-3466-4e19-85eb-c81977d57031)


- 결측값 없음
- 총 데이터 11162개
- y값 분포여부 확인

---

## autoML사용을 위한 setup

```python
s = setup(df, target ='deposit', session_id = 1, data_split_stratify=True, train_size= 0.9)
```

![image4](https://github.com/user-attachments/assets/21674786-118a-42c4-a895-0a95903e5c0e)


- y값 0, 1로 변환
- object형 변수 전처리
- train, test 데이터 분할 확인
- 교차검증 확인

## 분류 모델 성능 비교

```python
top5 = compare_models(n_select=5)
```

![image5](https://github.com/user-attachments/assets/dfc058f9-dc8f-4ff4-9e50-98f9243e93d5)


## 상위 5개 모델 확인

```python
top5
```

![image 6png](https://github.com/user-attachments/assets/6968ab01-ae12-40bc-bcc4-5bd749f69b7c)


## 상위 5개 모델 하이퍼파라미터 튜닝

```python
tuned_top5 = [tune_model(i) for i in top5]
for model in tuned_top5: # 각 모델에 대해 feature 중요도 시각화 수행
    print(f"Feature Importance - Model {type(model).__name__}")
    plot_model(model, plot='feature')
    plt.show()
```

- LGBMClassifier

- feature_importance

![image7](https://github.com/user-attachments/assets/dd1e5dba-86f7-48d0-84f8-ef84c3c5c556)


![image77](https://github.com/user-attachments/assets/7f6c3a17-c301-4e99-8440-1507ded9ae49)


- auc = 0.7943 → 0.7868
- rec = 0.7404 → 0.7282

- GradientBoostingClassifier

- feature_importance

![image8](https://github.com/user-attachments/assets/a78941c1-b357-4fcb-8638-7e09634a68fb)


![image88](https://github.com/user-attachments/assets/1e160023-a0ef-43de-912a-798a67e85c05)


- auc = 0.7894 → 0.7925
- rec = 0.7404 → 0.7354

- XGBClassifier

![image9](https://github.com/user-attachments/assets/42ba441c-d1d2-4b7b-b0c7-f2deeb1131e4)


- auc = 0.7843 → 0.7729
- rec = 0.7272 → 0.6948

- feature_importance

![image99](https://github.com/user-attachments/assets/b0b0dde1-cd5b-475a-b96d-4cbb075fb62d)


- RandomForestClassifier`

![image10](https://github.com/user-attachments/assets/531cfef2-2184-4bd5-ada9-3b9271ee4cb6)



- auc = 0.7792 → 0.7579
- rec = 0.7242 → 0.7044

- featdre_importance

![image100](https://github.com/user-attachments/assets/49befb81-8ae5-4ae7-ba4d-31d491ab27f8)


- AdaBoostClassifier

- feature_importance

![image11](https://github.com/user-attachments/assets/59ebb8d2-86d6-49cc-842d-f9beb1973801)


- auc = 0.7718 → 0.7397
- rec = 0.7138 → 0.6842

![image111](https://github.com/user-attachments/assets/41c393d1-8a14-499f-a970-06261daffae4)


⇒ **하이퍼 파라미터 튜닝하니까 전반적으로 성능 떨어짐**

## 튜닝된 5개 모델 확인

```python
tuned_top5
```

![image12](https://github.com/user-attachments/assets/e334bd48-e0d6-4057-a338-c32b93b523ab)


**⇒ 튜닝된 모델이 없음. 모두 기존 모델을 사용**

---

## 상위 5개 모델 앙상블

```python
blender_top5 = blend_models(estimator_list=tuned_top5)
```

![image13](https://github.com/user-attachments/assets/7298b4f3-77ea-4b01-acfa-0b0da007c8b6)


**⇒ 최고 성능 LGBM과 비교해서 AUC, rec 성능이 아주 조금 올라감**

---

## ROC curve 및 혼동행렬 확인

![image14](https://github.com/user-attachments/assets/17450fb8-9493-4dda-8879-c2b5e414811c)

![image15](https://github.com/user-attachments/assets/8f971ff4-70ed-469d-bada-7b34f2949079)

- **개선사항** : 전처리 과정을 거친 모델을 돌렸는데 성능이 낮아짐 (스케일링, 인코딩)
  - 전처리 없이 모델만 학습했더니 더 높은 성능을 보임
  - 사용한 모델(의사결정나무, 랜덤 포레스트)은 스케일링이나 인코딩의 영향을 덜 받음(특성의 스케일에 덜 민감)
 
- **트러블슈팅**
  - 발생 문제: import 과정에서 알 수 없는 모듈 오류 발생
    <p align="center">
    <img src="https://github.com/user-attachments/assets/51418da2-890d-4b27-b816-f7c54c83decd" width="50%" /> </p><br>
  - 발생 원인: 버전 충돌로 import가 안 됨
    - pycaret 내부에 설치되어 있는 패키지 버전이 충돌 
  - 해결 방법: 버전을 직접 설정해줌 
    ```python
    pip install scipy==1.9.3
    ```

- **개선하고 싶은 점** : 데이터 전처리 과정을 거친 후 학습했을 때 성능이 더 잘 나올 수 있는 방법을 탐색하고 싶음
  
