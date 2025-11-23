# 🟦 Sistema de Reconhecimento de Emoções Faciais

<div align="center">

**Sistema completo de reconhecimento de emoções faciais utilizando Deep Learning com CNN**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)](https://www.tensorflow.org/)

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

## 🟦 Como Usar

Execute o script principal:

```bash
python main.py
```

O sistema irá:
1. Carregar e pré-processar as imagens
2. Criar pasta de teste com timestamp: `Output/test_YYYYMMDD_HHMMSS/`
3. Treinar o modelo CNN por 30 épocas
4. Avaliar no conjunto de teste
5. Calcular métricas detalhadas
6. Gerar gráficos e salvar modelo
7. Gerar relatório Markdown completo

### Personalizar Parâmetros

Edite as constantes em `main.py`:

```python
EPOCHS = 30              # Número de épocas
BATCH_SIZE = 64          # Tamanho do batch
VALIDATION_SPLIT = 0.1   # 10% para validação
RANDOM_SEED = 42         # Seed para reprodutibilidade
TARGET_ACCURACY_MIN = 0.75  # 75%
TARGET_ACCURACY_MAX = 0.89  # 89%
```

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

Cada execução cria uma pasta única com:

```
Output/test_YYYYMMDD_HHMMSS/
├── docs/
│   └── test_report.md        # Relatório completo em Markdown
├── model_emotion_recognition.keras
├── training_history.png
├── accuracy.png
└── loss.png
```

**Conteúdo do Relatório:**
- Identificador do teste (timestamp, seed)
- Parâmetros de treinamento
- Informações do dataset
- Métricas principais
- Referências aos gráficos
- Data/hora de início e término
- Observações relevantes
- Objetivo de desempenho (75-89%)

---

## 🟦 Estrutura do Projeto

```
facial-recognition/
├── Dataset/FER-2013/          # Dataset organizado por emoções
├── Output/                     # Resultados e históricos
│   └── test_YYYYMMDD_HHMMSS/  # Pasta de cada teste
├── main.py                     # Script principal
├── requirements.txt            # Dependências
└── README.md
```

---

## 🟦 Tecnologias

- **TensorFlow/Keras**: Framework de Deep Learning
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
- **LinkedIn**: [Seu Perfil](https://www.linkedin.com/in/adriana-lima08/)