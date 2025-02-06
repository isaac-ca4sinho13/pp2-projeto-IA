import pandas as pd
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

#criando a variável do dataset
df = pd.read_csv('crop_yield_data.csv')

# renomeando as colunas:
df.rename(columns={
    'rainfall_mm': 'Chuva_mm',
    'soil_quality_index': 'indice_qualidade_solo',
    'farm_size_hectares': 'fazenda_tamanho_ha',
    'fertilizer_kg': 'fertilizante_kg',
    'sunlight_hours': 'Horas_sol',
    'crop_yield': 'rendimento'
}, inplace=True)



# Mostrando as 5 primeiras linhas do dataset
st.write("### Primeiras linhas do dataset:")
st.write(df.head())

# Verificando valores nulos
st.write("### Quantidade de valores nulos:")
st.write(df.isnull().sum())

# Criando gráficos no Streamlit
st.write("## 📊 Visualizações")

# Gráfico de dispersão: Horas de Sol vs Rendimento
st.write("### Horas de Sol vs Rendimento da Colheita")
fig, ax = plt.subplots()
ax.scatter(df['Horas_sol'], df['rendimento'], s=10, alpha=1)
ax.set_xlabel("Horas de Sol")
ax.set_ylabel("Rendimento da Colheita")
ax.set_title("Comparação entre Horas de Sol e Rendimento da Colheita")
st.pyplot(fig)

# Gráfico de barras: Horas de Sol vs Rendimento
st.write("### Comparação entre Horas de Sol e Rendimento da Colheita")
fig, ax = plt.subplots()
ax.bar(df['Horas_sol'], df['rendimento'])
ax.set_xlabel("Horas de Sol")
ax.set_ylabel("Rendimento da Colheita")
ax.set_title("Comparação entre Horas de Sol e Rendimento da Colheita")
st.pyplot(fig)

# Comparação entre Quantidade de Chuva e Qualidade do Solo
st.write("### Comparação entre Chuva e Qualidade do Solo")
fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(df['Chuva_mm'], df['indice_qualidade_solo'])
ax.set_ylabel("Qualidade do Solo")
ax.set_xlabel("Quantidade de Chuva")
ax.set_title("Comparação entre Chuva e Qualidade do Solo")
st.pyplot(fig)

# Índice de Qualidade do Solo vs Rendimento
st.write("### Comparação entre Qualidade do Solo e Rendimento")
fig, ax = plt.subplots()
ax.bar(df['indice_qualidade_solo'], df['rendimento'])
ax.set_xlabel("Índice de Qualidade do Solo")
ax.set_ylabel("Rendimento")
ax.set_title("Comparação entre Qualidade do Solo e Rendimento")
st.pyplot(fig)

# Comparação entre tamanho da fazenda e rendimento
st.write("### Comparação entre Tamanho da Fazenda e Rendimento")
fig, ax = plt.subplots()
ax.plot(df['fazenda_tamanho_ha'], df['rendimento'])
ax.set_xlabel("Tamanho da Fazenda (ha)")
ax.set_ylabel("Rendimento")
ax.set_title("Comparação entre Tamanho da Fazenda e Rendimento")
st.pyplot(fig)

# Horas de sol x fertilizante
st.write("### Horas de Sol vs Uso de Fertilizante")
fig, ax = plt.subplots()
hb = ax.hexbin(df['Horas_sol'], df['fertilizante_kg'], gridsize=30, cmap='coolwarm')
fig.colorbar(hb, label='Densidade')
ax.set_xlabel('Horas de Sol')
ax.set_ylabel('Fertilizante (kg)')
ax.set_title('Horas de Sol vs Uso de Fertilizante')
st.pyplot(fig)


# Variáveis independentes
X = df[['Horas_sol', 'fertilizante_kg', 'fazenda_tamanho_ha', 'indice_qualidade_solo', 'Chuva_mm']]

# Variável alvo
y = df['rendimento']

# Dividir os dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciar o modelo
modelo = LinearRegression()

# Treinar o modelo
modelo.fit(X_train, y_train)

# Fazer previsões com os dados de teste
y_pred = modelo.predict(X_test)

# Calcular as métricas de avaliação
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# --- Streamlit UI ---
st.title("Previsão de Rendimento da Colheita 🌾")

st.subheader("Métricas do Modelo")
st.write(f"✅ **RMSE:** {rmse:.4f}")
st.write(f"✅ **MAE:** {mae:.4f}")
st.write(f"✅ **R² Score:** {r2:.6f}")

st.subheader("Faça uma Previsão")

# Criar entradas para o usuário inserir valores
chuva_mm = st.number_input("Quantidade de chuva (mm):", min_value=0.0, step=0.1)
qualidade_solo = st.number_input("Índice de qualidade do solo:", min_value=0.0, max_value=10.0, step=0.1)
tamanho_fazenda = st.number_input("Tamanho da fazenda (hectares):", min_value=0.0, step=0.1)
fertilizante_kg = st.number_input("Quantidade de fertilizante (kg):", min_value=0.0, step=0.1)
horas_sol = st.number_input("Horas de sol por dia:", min_value=0.0, max_value=24.0, step=0.1)

# Botão para fazer a previsão
if st.button("Prever Rendimento"):
    # Criar um DataFrame com os dados inseridos
    entrada = pd.DataFrame([[horas_sol, fertilizante_kg, tamanho_fazenda, qualidade_solo, chuva_mm]],
                           columns=['Horas_sol', 'fertilizante_kg', 'fazenda_tamanho_ha', 'indice_qualidade_solo', 'Chuva_mm'])

    # Fazer a previsão
    modelo = joblib.load("modelo_regressao.pkl")
    previsao = modelo.predict(entrada)

    # Exibir a previsão
    st.success(f"🌾 **Previsão do rendimento da colheita:** {previsao[0]:.2f} toneladas")

