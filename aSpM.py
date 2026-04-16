import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data={

    "Experience":[1,2,3,4,5,6,7,8],
    "Degree":[1,1,2,2,3,3,4,4],
    "salary":[20000,25000,30000,35000,40000,45000,50000,55000]
}
df=pd.DataFrame(data)
X=df[["Experience", "Degree"]]
y=df["salary"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
AI_model=LinearRegression()
AI_model.fit(X_train,y_train)
predication=AI_model.predict(X_test)
print(predication)
salary=AI_model.predict(pd.DataFrame([[5,3]],columns=["Experience","Degree"]))
print(salary)
Accuracy=AI_model.score(X_test,y_test)
print(Accuracy)
