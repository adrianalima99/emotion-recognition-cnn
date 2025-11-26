# 🟦 Sistema de Reconhecimento de Emoções Faciais

<div align="center">

**Sistema completo de reconhecimento de emoções faciais utilizando Deep Learning com CNN**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

</div>

---

## 🟦 Sobre o Projeto

Sistema de **reconhecimento de emoções faciais** utilizando **Deep Learning** e **Redes Neurais Convolucionais (CNN)**. Identifica e classifica 7 emoções: raiva, nojo, medo, felicidade, tristeza, surpresa e neutro.

O projeto inclui um **sistema automatizado de histórico de testes** que gera relatórios Markdown detalhados com métricas, hiperparâmetros e análises de desempenho de cada treinamento.

### 🟦 Características Principais

- 🟦 **Modelo CNN**: Arquitetura otimizada para classificação de emoções
- 🟦 **Dataset FER-2013**: ~35.000 imagens organizadas
- 🟦 **Histórico Automático**: Relatórios Markdown gerados automaticamente
- 🟦 **Métricas Completas**: Accuracy, precision, recall, F1-score por classe
- 🟦 **Objetivo de Desempenho**: Meta de acurácia entre 75% e 89%
- 🟦 **Organização**: Pastas automáticas por timestamp para cada teste

### 🆕 Novidades da v2.0

- ✅ **Reconhecimento em tempo real via webcam**
- ✅ **Dois detectores de face**: Haar Cascade e DNN
- ✅ **Suavização de predições** (evita flickering)
- ✅ **Captura de snapshots** (manual e automática)
- ✅ **Log de emoções em CSV**
- ✅ **Relatório de sessão em Markdown**
- ✅ **Exibição de FPS em tempo real**
- ✅ **Controles interativos durante execução**

---

## 🟦 Instalação

### 1. Clone o Repositório

```bash
git clone https://github.com/seu-usuario/facial-recognition.git
cd facial-recognition
```

### 2. Ambiente Virtual

**Windows:**
```bash
python -m venv venv310
venv310\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv310
source venv310/bin/activate
```

### 3. Dependências

```bash
pip install -r requirements.txt
```

### 4. Estrutura do Dataset

Organize o dataset FER-2013 na seguinte estrutura:

```
Dataset/FER-2013/
├── train/
│   ├── angry/ ├── disgust/ ├── fear/ ├── happy/
│   ├── neutral/ ├── sad/ └── surprise/
└── test/
    ├── angry/ ├── disgust/ ├── fear/ ├── happy/
    ├── neutral/ ├── sad/ └── surprise/
```

---

---

## 🟦 Dataset

**FER-2013 (Facial Expression Recognition 2013)**

- **Total**: ~35.000 imagens (48x48 pixels, escala de cinza)
- **Treino**: ~28.000 imagens
- **Teste**: ~6.000 imagens
- **Classes**: 7 emoções básicas

---

## 🟦 Arquitetura do Modelo

```
Input: (48, 48, 1)
1. Conv2D: 32 filtros (3x3) + ReLU
2. MaxPooling2D: (2x2)
3. Conv2D: 64 filtros (3x3) + ReLU
4. MaxPooling2D: (2x2)
5. Flatten
6. Dense: 128 neurônios + ReLU
7. Dropout: 50%
8. Dense: 7 neurônios + Softmax

Total de Parâmetros: ~839.000
```

**Hiperparâmetros:** Optimizer: Adam | Loss: Categorical Crossentropy | Épocas: 30 | Batch Size: 64

---

## 🟦 Resultados

**Objetivo de Desempenho:** Acurácia entre 75% e 89%

**Métricas Calculadas:**
- Accuracy, Loss
- Precision, Recall, F1-Score (Macro e Weighted)
- Métricas por classe
- Matriz de Confusão

**Visualizações:** training_history.png, accuracy.png, loss.png

---

## 🟦 Sistema de Histórico

### Histórico de Treinamento

```
Output/test_YYYYMMDD_HHMMSS/
├── docs/
│   └── test_report.md        # Relatório completo em Markdown
├── model_emotion_recognition.keras
├── training_history.png
├── accuracy.png
└── loss.png
```

### 🆕 Histórico de Sessões Webcam

```
Output/webcam_session_YYYYMMDD_HHMMSS/
├── session_report.md    # Relatório Markdown da sessão
├── emotions_log.csv     # Log de todas as predições
└── snapshots/           # Imagens capturadas
    ├── snapshot_000030.jpg
    └── manual_143022.jpg
```

#### Conteúdo do emotions_log.csv

| Coluna | Descrição |
|--------|-----------|
| timestamp | Data/hora da predição |
| frame | Número do frame |
| emotion | Emoção detectada (EN) |
| emotion_pt | Emoção detectada (PT) |
| confidence | Confiança da predição |
| prob_angry...prob_neutral | Probabilidades por classe |

---

## 🟦 Estrutura do Projeto

```
emotion-recognition-cnn/
├── Dataset/FER-2013/              # Dataset organizado por emoções
├── Output/
│   ├── test_YYYYMMDD_HHMMSS/      # Pasta de cada treino
│   └── webcam_session_YYYYMMDD_HHMMSS/  # 🆕 Sessões de webcam
├── docs/                          # 🆕 Documentação de versões
│   ├── v1.0_relatorio.md
│   └── v2.0_relatorio.md
├── main.py                        # Script de treinamento
├── camera_demo.py                 # 🆕 Demo webcam em tempo real
├── inference_utils.py             # 🆕 Utilitários de inferência
├── requirements.txt
└── README.md
```

---

## 🟦 Tecnologias

- **TensorFlow/Keras**: Framework de Deep Learning
- **OpenCV**: 🆕 Captura de vídeo, detecção facial, visualização
- **NumPy**: Computação numérica
- **scikit-learn**: Métricas e utilitários
- **Matplotlib**: Visualizações
- **Pillow**: Processamento de imagens

---

## 🟦 Sobre a Desenvolvedora

- **Área de atuação**: Engenharia de Software, Front-End e Dados
- **Foco**: Dados, Deep Learning.
- **Habilidades**: Python, TensorFlow/Keras, Machine Learning.
- **Contato**: adriana.slima0899@gmail.com
- **LinkedIn**: [Adriana Lima](https://www.linkedin.com/in/adriana-lima08/)
