import pandas as pd
import numpy as np

from regression_tree import predict_age
#PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# lê os arquivos
df_train = pd.read_csv('train.csv',
    usecols=['Survived','Pclass','Sex', 'Age','SibSp','Parch','Embarked'])

df_test = pd.read_csv('test.csv',
    usecols=['Pclass','Sex', 'Age','SibSp','Parch','Embarked'])

df_all = [df_train, df_test]

# remove campos desnecessários
index = 0
for df in df_all:

    # Insere os valores que estão faltando em Age
    """ Versão 1 - usa árvore de regressão """
        #predict_age(df_train)

    """ Versão 2 - gera valores entre (mean — std) e (mean + std) """
    mean_age = df['Age'].mean()
    std_age = df['Age'].std() # desvio padrão

    min_value = mean_age - std_age
    max_value = mean_age + std_age

    size = df['Age'].isnull().sum()
    df.loc[df['Age'].isnull(), 'Age'] = np.random.randint(min_value, max_value, size=size)
    df['Age'] = df['Age'].astype(int)

    # Insere os valores faltando em Embarked
    embarked_values = ['C', 'Q', 'S']
    size = df['Embarked'].isnull().sum()
    random_values = [embarked_values[i] for i in np.random.randint(0, 3, size=size)]
    df.loc[df['Embarked'].isnull(), 'Embarked'] = random_values

    # Faz o one hot encode em Sex, Embaked, SibSp e Parch, Pclass
    # o  parâmetro'categories' garante que mesmo que o dado esteja faltando será considerado
    df['Sex'] = df['Sex'].astype('category', categories=['male', 'famale'])
    df['Embarked'] = df['Embarked'].astype('category', categories=embarked_values)
    df['Parch'] = df['Parch'].astype('category', categories=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    df['SibSp'] = df['SibSp'].astype('category', categories=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    df['Pclass'] = df['Pclass'].astype('category', categories=[1, 2, 3])
    
    res = pd.concat([df,pd.get_dummies(df['Sex'], prefix='Sex')], axis=1)
    res = pd.concat([res,pd.get_dummies(df['Parch'], prefix='Parch')], axis=1)
    res = pd.concat([res,pd.get_dummies(df['SibSp'], prefix='SibSp')], axis=1)
    res = pd.concat([res,pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)
    res = pd.concat([res,pd.get_dummies(df['Pclass'], prefix='Pclass')], axis=1)
    
    res.drop(['Sex', 'Embarked', 'Pclass', 'Parch', 'SibSp'], inplace=True, axis='columns')

    df_all[index] = res
    index +=1

# atualiza as referências
df_train = df_all[0]
df_test = df_all[1]

# imprime o impacto das colunas com a coluna Survived
# def print_impact(col_name):
#     res = df_train[[col_name, 'Survived']].groupby([col_name], as_index=False).mean()
#     print(res)

# Separa os dados
train_x = df_train.drop(['Survived'], inplace=False, axis='columns').values.tolist()
train_y = df_train['Survived'].values.tolist()

# exibe os dados do dataframe
print(df_train.head(10))

# exibe os dados para treino
for i in range(10):
    print(train_x[i])