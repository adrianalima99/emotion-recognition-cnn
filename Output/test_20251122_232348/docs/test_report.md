# Relatório de Teste - Reconhecimento de Emoções Faciais

## 📋 Identificador do Teste

- **ID do Teste**: `test_20251122_232348`
- **Data/Hora de Início**: 22/11/2025 23:23:48
- **Data/Hora de Término**: 22/11/2025 23:36:57
- **Duração do Treinamento**: 0h 13min 9s
- **Random Seed**: 42

---

## 🎯 Objetivo de Desempenho

**Objetivo de acurácia: entre 75% e 89%.**

Acurácia obtida neste teste: **52.06%**

⚠️ **Status**: Acurácia abaixo do objetivo mínimo (75.0%)

---

## 📊 Parâmetros de Treinamento

### Hiperparâmetros
- **Épocas (Epochs)**: 30
- **Batch Size**: 64
- **Validation Split**: 10.0%
- **Optimizer**: adam
- **Loss Function**: categorical_crossentropy
- **Random Seed**: 42

### Arquitetura do Modelo
- **Tipo**: CNN (Convolutional Neural Network)
- **Input Shape**: (48, 48, 1)
- **Total de Parâmetros**: 839,047

#### Estrutura das Camadas:
1. **Conv2D**: 32 filtros, kernel (3, 3), ativação relu
2. **MaxPooling2D**: Pool size (2, 2)
3. **Conv2D**: 64 filtros, kernel (3, 3), ativação relu
4. **MaxPooling2D**: Pool size (2, 2)
5. **Flatten**
6. **Dense**: 128 neurônios, ativação relu
7. **Dropout**: Taxa 50.0%
8. **Dense**: 7 neurônios, ativação softmax


---

## 📁 Informações sobre o Dataset

### Dataset Utilizado
- **Nome**: FER-2013 (Facial Expression Recognition 2013)
- **Formato**: Imagens JPG organizadas por pastas de emoções
- **Resolução**: 48x48 pixels (escala de cinza)

### Divisão dos Dados
- **Treino**: 28,709 imagens
  - Validação (10% do treino): ~2,870 imagens
  - Treino efetivo: ~25,839 imagens
- **Teste**: 7,178 imagens

### Distribuição por Emoção - Treino
- **Angry**: 3,995 imagens (13.9%)
- **Disgust**: 436 imagens (1.5%)
- **Fear**: 4,097 imagens (14.3%)
- **Happy**: 7,215 imagens (25.1%)
- **Sad**: 4,830 imagens (16.8%)
- **Surprise**: 3,171 imagens (11.0%)
- **Neutral**: 4,965 imagens (17.3%)

### Distribuição por Emoção - Teste
- **Angry**: 958 imagens (13.3%)
- **Disgust**: 111 imagens (1.5%)
- **Fear**: 1,024 imagens (14.3%)
- **Happy**: 1,774 imagens (24.7%)
- **Sad**: 1,247 imagens (17.4%)
- **Surprise**: 831 imagens (11.6%)
- **Neutral**: 1,233 imagens (17.2%)


---

## 📈 Métricas Principais

### Métricas Gerais
- **Acurácia (Accuracy)**: 52.06%
- **Loss**: 1.6296
- **Precision (Macro)**: 55.26%
- **Recall (Macro)**: 46.41%
- **F1-Score (Macro)**: 47.87%
- **Precision (Weighted)**: 52.85%
- **Recall (Weighted)**: 52.06%
- **F1-Score (Weighted)**: 50.93%

### Métricas por Classe

#### Angry
- **Precision**: 40.13%
- **Recall**: 39.87%
- **F1-Score**: 40.00%
- **Support**: 958 amostras

#### Disgust
- **Precision**: 73.53%
- **Recall**: 22.52%
- **F1-Score**: 34.48%
- **Support**: 111 amostras

#### Fear
- **Precision**: 40.56%
- **Recall**: 36.72%
- **F1-Score**: 38.54%
- **Support**: 1024 amostras

#### Happy
- **Precision**: 62.88%
- **Recall**: 80.89%
- **F1-Score**: 70.76%
- **Support**: 1774 amostras

#### Sad
- **Precision**: 37.42%
- **Recall**: 48.92%
- **F1-Score**: 42.41%
- **Support**: 1247 amostras

#### Surprise
- **Precision**: 75.07%
- **Recall**: 68.11%
- **F1-Score**: 71.42%
- **Support**: 831 amostras

#### Neutral
- **Precision**: 57.26%
- **Recall**: 27.82%
- **F1-Score**: 37.45%
- **Support**: 1233 amostras

### Evolução durante o Treinamento
- **Acurácia Final (Treino)**: 71.80%
- **Acurácia Final (Validação)**: 27.55%
- **Loss Final (Treino)**: 0.7029
- **Loss Final (Validação)**: 2.7972


---

## 📸 Gráficos e Visualizações

Os seguintes gráficos foram gerados e salvos nesta pasta:

1. **training_history.png**: Gráfico combinado de accuracy e loss
2. **accuracy.png**: Gráfico detalhado de accuracy (treino e validação)
3. **loss.png**: Gráfico detalhado de loss (treino e validação)

### Localização dos Arquivos:
- `Output\test_20251122_232348/training_history.png`
- `Output\test_20251122_232348/accuracy.png`
- `Output\test_20251122_232348/loss.png`

---

## 📝 Observações e Logs

### Análise do Treinamento
- ⚠️ **Possível overfitting detectado**: Diferença significativa entre acurácia de treino e validação.
- ⚠️ **Validação loss maior que treino**: Modelo pode estar se adaptando demais aos dados de treino.
- ℹ️ **Acurácia abaixo do objetivo**: Modelo alcançou 52.06%, objetivo é 75-89%.


### Histórico de Épocas

#### Últimas 5 Épocas:

| Época | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| 26 | 69.48% | 26.82% | 0.7708 | 2.5421 |
| 27 | 70.15% | 27.38% | 0.7489 | 2.5090 |
| 28 | 70.76% | 22.95% | 0.7359 | 2.8158 |
| 29 | 70.84% | 28.07% | 0.7178 | 2.5738 |
| 30 | 71.80% | 27.55% | 0.7029 | 2.7972 |



---

## 💾 Arquivos do Teste

- **Modelo Salvo**: `model_emotion_recognition.keras` (raiz do projeto)
- **Relatório**: Este arquivo (`test_report.md`)
- **Gráficos**: Pasta raiz deste teste

---

**Relatório gerado automaticamente em**: 22/11/2025 23:46:10
