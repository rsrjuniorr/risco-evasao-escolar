import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# CONFIGURAÇÃO DA PÁGINA E DESIGN (CSS)
# ==========================================
st.set_page_config(
    page_title="Risco de Evasão Escolar", page_icon="🎓", layout="centered"
)

# Estilo visual limpo: fundo cinza claro moderno e textos escuros legíveis
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F8FAFC;
    }
    body, .stMarkdown, .stText, p, span, label {
        color: #1E293B !important;
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
faltas = np.random.randint(0, 25, n_linhas)  # Até 25 faltas para abranger o limite de 20
media_notas = np.random.uniform(3.0, 10.0, n_linhas)
atividades_entregues = np.random.randint(0, 5, n_linhas)  # De 0 a 4 atividades

evasao = []
for i in range(n_linhas):
  # Regras institucionais aplicadas na geração dos dados de treino
  if faltas[i] >= 20 or atividades_entregues[i] < 1 or media_notas[i] < 7.0:
    evasao.append(1)  # Maior propensão a risco/problema acadêmico
  else:
    pontuacao = (
        (faltas[i] * 1.0)
        - (media_notas[i] * 1.5)
        - (horas_estudo[i] * 0.4)
        - (atividades_entregues[i] * 0.6)
    )
    if pontuacao > 1.0:
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
# PASSO 3: INTERFACE DO SITE (Streamlit)
# ==========================================

st.markdown(
    "<h1 style='text-align: center; color: #1E3A8A;'>🎓 Monitor de Risco e"
    " Desempenho Acadêmico</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #4B5563;'>Sistema integrado com"
    " regras institucionais e Árvore de Decisão para acompanhamento"
    " acadêmico.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

st.subheader("📝 Informe os Dados do Aluno")
st.write(
    "Ajuste os parâmetros abaixo para verificar a situação acadêmica e o"
    " risco:"
)

col1, col2 = st.columns(2)

with col1:
  horas = st.slider("Horas de estudo / semana", 1, 40, 15)
  # Nota ajustada de 0 a 10 (onde 7.0 equivale a 70 pontos)
  media = st.slider("Média de notas (0 a 10)", 0.0, 10.0, 7.5)

with col2:
  # Faltas considerando o limite máximo de 80 aulas
  faltas = st.slider("Número de faltas (máx. 80)", 0, 80, 5)
  # 4 atividades exigidas no semestre
  atividades = st.slider(
      "Atividades entregues (de 0 a 4)", 0, 4, 4
  )

st.markdown("")

analisar = st.button(
    "🔍 Analisar Situação Acadêmica", use_container_width=True, type="primary"
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
  st.subheader("📊 Resultado e Validação de Regras")

  # Validação das Regras Institucionais Rígidas
  reprovado_por_falta = faltas >= 20
  reprovado_por_atividade = atividades < 1
  deve_fazer_final = media < 7.0

  # Exibição de Alertas Específicos baseados nas regras da faculdade
  if reprovado_por_falta:
    st.error(
        f"❌ **REPROVADO POR FALTAS**\n\nO aluno possui **{faltas} faltas**"
        f" (o limite máximo tolerado é de 19 faltas em 80 aulas)."
    )
  elif reprovado_por_atividade:
    st.error(
        f"❌ **REPROVADO POR FALTA DE ATIVIDADES**\n\nO aluno entregou apenas"
        f" **{atividades} atividade(s)**. É obrigatório entregar pelo menos 1"
        f" atividade."
    )
  else:
    # Se passou nas regras rígidas, avalia a nota e o modelo de árvore
    if deve_fazer_final:
      st.warning(
          f"⚠️ **ATENÇÃO: ALUNO DE SUA 70 (PRECISA DE PROVA FINAL)**\n\n"
          f"A média do aluno é **{media*10:.0f} de 100** (abaixo de 70). O"
          f" aluno está obrigado a realizar a prova final."
      )
    elif predicao == 1:
      st.warning(
          f"⚠️ **ALERTA DE ALTO RISCO DE EVASÃO**\n\nO perfil acadêmico aponta"
          f" vulnerabilidade e risco de evasão (Probabilidade:"
          f" {probabilidade[1]*100:.1f}%)."
      )
    else:
      st.success(
          f"✅ **ALUNO APROVADO / ESTÁVEL**\n\n"
          f"Média satisfatória, frequência em dia e entregas realizadas com"
          f" sucesso (Probabilidade de estabilidade:"
          f" {probabilidade[0]*100:.1f}%)."
      )

  # Explicar o caminho lógico da árvore
  st.markdown("### 🌳 Caminho da Decisão do Modelo (Machine Learning)")
  st.info(
      "Além das regras institucionais acima, a Árvore de Decisão avaliou o"
      " perfil comportamental com base nestes critérios:"
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
    - **Regras Institucionais Incorporadas:** Limite de 20 faltas (de 80 aulas), entrega obrigatória de atividades e nota de corte 70 para dispensa de final.
    - **Técnica de Mineração:** Árvore de Decisão (`DecisionTreeClassifier` do scikit-learn).
    - **Objetivo Acadêmico:** Unir regras determinísticas de secretaria com inteligência preditiva de Machine Learning.
    """)
