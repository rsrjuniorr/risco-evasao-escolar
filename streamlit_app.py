import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# PASSO 1: GERAÇÃO DE DADOS EQUILIBRADA
# ==========================================
# Mudamos o arquivo para forçar a regerar os dados com melhor distribuição
arquivo_dados = "dados_evasao.csv"
arquivo_modelo = "modelo_evasao.pkl"

# Se quiser forçar a recriação com dados mais balanceados:
if os.path.exists(arquivo_dados):
  os.remove(arquivo_dados)
if os.path.exists(arquivo_modelo):
  os.remove(arquivo_modelo)

np.random.seed(10)  # Nova semente para melhor distribuição
n_linhas = 400

horas_estudo = np.random.randint(2, 35, n_linhas)
faltas = np.random.randint(0, 20, n_linhas)
media_notas = np.random.uniform(3.0, 10.0, n_linhas)
atividades_entregues = np.random.randint(1, 10, n_linhas)

evasao = []
for i in range(n_linhas):
  # Fórmula balanceada para dar variação real de risco
  pontuacao = (
      (faltas[i] * 2.0)
      - (media_notas[i] * 1.5)
      - (horas_estudo[i] * 0.4)
      - (atividades_entregues[i] * 0.6)
  )
  ruido = np.random.normal(
      0, 3
  )  # Ruído controlado para misturar os dados realisticamente
  total = pontuacao + ruido

  # Ponto de corte ajustado para balancear as classes (~50% cada)
  if total > 2.5:
    evasao.append(1)  # Alto Risco
  else:
    evasao.append(0)  # Baixo Risco

df = pd.DataFrame(
    {
        "horas_estudo_semana": horas_estudo,
        "faltas": faltas,
        "media_notas": np.round(media_notas, 1),
        "atividades_entregues": atividades_entregues,
        "evadiu": evasao,
    }
)
df.to_csv(arquivo_dados, index=False)

# ==========================================
# PASSO 2: TREINAMENTO DO MODELO
# ==========================================
X = df[["horas_estudo_semana", "faltas", "media_notas", "atividades_entregues"]]
y = df["evadiu"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# max_depth=4 para permitir um aprendizado um pouco mais rico nas regras
modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
modelo.fit(X_train, y_train)

joblib.dump(modelo, arquivo_modelo)

# ==========================================
# PASSO 3: INTERFACE DO SITE (Streamlit)
# ==========================================
st.title("🎓 Monitor de Risco de Evasão Escolar")
st.write(
    "Esta aplicação utiliza uma **Árvore de Decisão** para estimar o risco de"
    " um aluno evadir com base no seu perfil acadêmico."
)

# Formulário lateral
st.sidebar.header("Perfil do Aluno")
horas = st.sidebar.slider("Horas de estudo por semana", 1, 40, 10)
faltas = st.sidebar.slider("Número de faltas", 0, 25, 8)
media = st.sidebar.slider("Média de notas (0 a 10)", 0.0, 10.0, 6.0)
atividades = st.sidebar.slider(
    "Atividades entregues (último mês)", 0, 10, 5
)

if st.button("Analisar Risco"):
  dados_entrada = pd.DataFrame(
      [[horas, faltas, media, atividades]],
      columns=[
          "horas_estudo_semana",
          "faltas",
          "media_notas",
          "atividades_entregues",
      ],
  )

  predicao = modelo.predict(dados_entrada)[0]
  probabilidade = modelo.predict_proba(dados_entrada)[0]

  st.subheader("Resultado da Análise")
  if predicao == 1:
    st.error(
        f"⚠️ **Alto Risco de Evasão** (Probabilidade de evasão:"
        f" {probabilidade[1]*100:.1f}%)"
    )
  else:
    st.success(
        f"✅ **Baixo Risco / Aluno Estável** (Probabilidade de permanência:"
        f" {probabilidade[0]*100:.1f}%)"
    )

  # Explicar o caminho lógico da árvore
  st.markdown("---")
  st.subheader("🌳 Caminho da Árvore de Decisão (Regras percorridas)")
  node_indicator = modelo.decision_path(dados_entrada)
  leaf_id = modelo.apply(dados_entrada)
  feature = modelo.tree_.feature
  threshold = modelo.tree_.threshold

  sample_id = 0
  node_index = node_indicator.indices[
      node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
  ]

  feature_names = [
      "horas_estudo_semana",
      "faltas",
      "media_notas",
      "atividades_entregues",
  ]
  for node_id in node_index:
    if leaf_id[sample_id] == node_id:
      continue
    feature_id = feature[node_id]
    threshold_val = threshold[node_id]
    val_usuario = dados_entrada.iloc[0, feature_id]

    if val_usuario <= threshold_val:
      st.write(
          f"- O aluno tem **{feature_names[feature_id]}** = `{val_usuario}`"
          f" (menor ou igual a `{threshold_val:.1f}`)"
      )
    else:
      st.write(
          f"- O aluno tem **{feature_names[feature_id]}** = `{val_usuario}`"
          f" (maior que `{threshold_val:.1f}`)"
      )

# Seção Como Funciona
st.markdown("---")
with st.expander("ℹ️ Como Funciona (Detalhes Técnicos)"):
  st.write("""
    - **Técnica:** Árvore de Decisão (`DecisionTreeClassifier`).
    - **Dataset:** 400 linhas geradas sinteticamente via script Python com regras e ruído controlado.
    - **Métrica:** Acurácia no conjunto de teste.
    """)
