import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Carregar o conjunto de dados
df = pd.read_csv("data.csv")

# Separar as variáveis independentes (X) da variável dependente (y)
X = df.drop("target_variable", axis=1)
y = df["target_variable"]

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Treinar o modelo de regressão linear
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Fazer previsões nos dados de teste
y_pred = regressor.predict(X_test)

# Avaliar o desempenho do modelo
error = np.mean((y_test - y_pred)**2)
print("Erro quadrático médio: ", error)
