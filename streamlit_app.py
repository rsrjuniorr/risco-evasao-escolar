import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# CONFIGURAÇÃO DA PÁGINA (Layout limpo em coluna única)
# ==========================================
st.set_page_config(
    page_title="Risco de Evasão Escolar", page_icon="🎓", layout="centered"
)

# Estilo de fundo opcional (Cinza/gelo bem moderno)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F8FAFC;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# PASSO 1: GERAÇÃO DE DADOS EQUILIBRADA
# ==========================================
arquivo_dados = "dados_evasao.csv"
arquivo_modelo = "modelo_evasao.pkl"

if os.path.exists(arquivo_dados):
  os.remove(arquivo_dados)
if os.path.exists(arquivo_modelo):
  os.remove(arquivo_modelo)

np.random.seed(10)
n_linhas = 400

horas_estudo = np.random.randint(2, 35, n_linhas)
faltas = np.random.randint(0, 20, n_linhas)
media_notas = np.random.uniform(3.0, 10.0, n_linhas)
atividades_entregues = np.random.randint(1, 10, n_linhas)

evasao = []
for i in range(n_linhas):
  pontuacao = (
      (faltas[i] * 2.0)
      - (media_notas[i] * 1.5)
      - (horas_estudo[i] * 0.4)
      - (atividades_entregues[i] * 0.6)
  )
  ruido = np.random.normal(0, 3)
  total = pontuacao + ruido

  if total > 2.5:
    evasao.append(1)
  else:
    evasao.append(0)

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
modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
modelo.fit(X_train, y_train)
joblib.dump(modelo, arquivo_modelo)

# ==========================================
# PASSO 3: INTERFACE DO SITE (Design Limpo)
# ==========================================

st.markdown(
    "<h1 style='text-align: center; color: #1E3A8A;'>🎓 Monitor de Risco de"
    " Evasão Escolar</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #4B5563;'>Sistema inteligente de"
    " apoio pedagógico baseado em Árvore de Decisão para prever e prevenir a"
    " evasão de alunos.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

st.subheader("📝 Informe os Dados do Aluno")
st.write(
    "Arraste os seletores abaixo com o perfil acadêmico que deseja analisar:"
)

col1, col2 = st.columns(2)

with col1:
  horas = st.slider("Horas de estudo / semana", 1, 40, 15)
  media = st.slider("Média de notas (0 a 10)", 0.0, 10.0, 7.0)

with col2:
  faltas = st.slider("Número de faltas", 0, 25, 4)
  atividades = st.slider("Atividades entregues", 0, 10, 8)

st.markdown("")

analisar = st.button(
    "🔍 Analisar Risco Acadêmico", use_container_width=True, type="primary"
)

if analisar:
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

  st.markdown("---")
  st.subheader("📊 Resultado da Análise")

  if predicao == 1:
    st.error(
        f"⚠️ **ALTO RISCO DE EVASÃO DETECTADO**\n\n"
        f"- Probabilidade de Evasão: **{probabilidade[1]*100:.1f}**%\n"
        f"- *Recomendação:* O aluno apresenta fatores críticos (como faltas"
        f" elevadas ou baixo rendimento). Recomenda-se abordagem de"
        f" acompanhamento pedagógico."
    )
  else:
    st.success(
        f"✅ **BAIXO RISCO / ALUNO ESTÁVEL**\n\n"
        f"- Probabilidade de Permanência: **{probabilidade[0]*100:.1f}**%\n"
        f"- *Recomendação:* O perfil indica bom engajamento acadêmico e"
        f" estabilidade."
    )

  st.markdown("### 🌳 Caminho da Decisão (Regras do Modelo)")
  st.info(
      "O modelo de Machine Learning chegou a esta conclusão percorrendo as"
      " seguintes regras com base nos dados informados:"
  )

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
          f"🔹 **{feature_names[feature_id]}** (`{val_usuario}`) é menor ou"
          f" igual a `{threshold_val:.1f}`"
      )
    else:
      st.write(
          f"🔹 **{feature_names[feature_id]}** (`{val_usuario}`) é maior que"
          f" `{threshold_val:.1f}`"
      )

st.markdown("---")
with st.expander("ℹ️ Sobre o Sistema e Detalhes Técnicos"):
  st.write("""
    - **Técnica de Mineração:** Árvore de Decisão (`DecisionTreeClassifier` do scikit-learn).
    - **Base de Dados:** 400 registros simulados via script com modelagem de correlação realista e ruído controlado.
    - **Objetivo Acadêmico:** Atender aos requisitos do projeto prático de Mineração de Dados, garantindo interpretabilidade total das regras de decisão.
    """)
