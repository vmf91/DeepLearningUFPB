import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd

def predict_age(df: pd.DataFrame )->pd.DataFrame:
    # remove a coluna Cabin
    aux = df.drop(['Embarked'], inplace=False, axis=1)

    # remove as linhas em que Age ou Embarked (poucas colunas estão com o Embarked faltando)
    train = aux.dropna(subset=['Age'], inplace=False)

    # separa a coluna Age das outras para o treinamento
    x = train.drop(['Age'], inplace=False, axis=1).values.tolist()
    y = train['Age'].values.tolist()

    # Fit regression model
    regr = DecisionTreeRegressor(max_depth=5)    
    regr.fit(x, y)

    # salva as linhas que estão sem a idade
    data_to_predict = []
    indexes = []
    for i in range(len(df['Age'])):
        if str(df.loc[i, 'Age']) == 'nan':
            indexes.append(i)
            data_to_predict.append([df.loc[i, 'Survived'], df.loc[i, 'Pclass'], 
                df.loc[i, 'Sex'], df.loc[i, 'SibSp'], df.loc[i, 'Parch'], df.loc[i, 'Fare']])       

    
    # calcula as idades e atribui
    result = regr.predict(data_to_predict)
    result = [round(res) for res in result]
    
    for i in range(len(result)):
        df.loc[indexes[i], 'Age'] = result[i]
        
    return df