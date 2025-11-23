import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from PIL import Image


# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================

# Configurações do Dataset
DATASET_PATH = 'Dataset/FER-2013'
TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
TEST_PATH = os.path.join(DATASET_PATH, 'test')

# Configurações de Treinamento
EPOCHS = 30
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42

# Objetivo de Acuracia
TARGET_ACCURACY_MIN = 0.75
TARGET_ACCURACY_MAX = 0.89

# Mapeamento de emoções para números
EMOTION_MAP = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}

# Lista de emoções na ordem correta
EMOTION_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


# ============================================================================
# FUNÇÕES DE UTILIDADE
# ============================================================================

def create_test_directory():
    """
    Cria uma nova pasta de teste baseada em timestamp.
    
    Returns:
        test_dir: Caminho completo da pasta do teste (ex: Output/test_20250115_143022/)
        docs_dir: Caminho completo da pasta docs dentro do teste
    """
    # Criar nome da pasta baseado em timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    test_dir = os.path.join('Output', f'test_{timestamp}')
    docs_dir = os.path.join(test_dir, 'docs')
    
    # Criar diretórios
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    print(f'\n{"="*60}')
    print(f'Pasta do teste criada: {test_dir}')
    print(f'{"="*60}\n')
    
    return test_dir, docs_dir


def load_images_from_directory(directory_path):
    """
    Carrega imagens de um diretório organizado por pastas de emoções.
    
    Args:
        directory_path: Caminho para o diretório contendo subpastas de emoções
        
    Returns:
        images: Array numpy com as imagens (N, 48, 48, 1)
        labels: Array numpy com os labels (N,)
        stats: Dicionário com estatísticas de carregamento
    """
    images = []
    labels = []
    stats = {}
    
    # Iterar sobre cada pasta de emoção
    for emotion_name, emotion_label in EMOTION_MAP.items():
        emotion_path = os.path.join(directory_path, emotion_name)
        
        if not os.path.exists(emotion_path):
            print(f'Aviso: Pasta {emotion_path} não encontrada. Pulando...')
            stats[emotion_name] = 0
            continue
        
        # Contador de imagens carregadas
        loaded_count = 0
        
        # Listar todos os arquivos JPG na pasta
        for filename in os.listdir(emotion_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(emotion_path, filename)
                
                try:
                    # Carregar imagem usando PIL
                    img = Image.open(img_path)
                    
                    # Converter para escala de cinza se necessário
                    if img.mode != 'L':
                        img = img.convert('L')
                    
                    # Redimensionar para 48x48 se necessário
                    img = img.resize((48, 48), Image.Resampling.LANCZOS)
                    
                    # Converter para array numpy
                    img_array = np.array(img, dtype=np.float32)
                    
                    # Normalizar para o range [0, 1]
                    img_array = img_array / 255.0
                    
                    # Adicionar dimensão do canal
                    img_array = np.expand_dims(img_array, axis=-1)
                    
                    images.append(img_array)
                    labels.append(emotion_label)
                    loaded_count += 1
                    
                except Exception as e:
                    print(f'Erro ao carregar {img_path}: {e}')
                    continue
        
        stats[emotion_name] = loaded_count
        print(f'Carregadas {loaded_count} imagens da categoria: {emotion_name}')
    
    # Converter para arrays numpy
    images = np.array(images)
    labels = np.array(labels)
    
    print(f'\nTotal de imagens carregadas: {len(images)}')
    print(f'Formato das imagens: {images.shape}')
    print(f'Formato dos labels: {labels.shape}')
    
    return images, labels, stats


def load_dataset():
    """
    Carrega o dataset completo (treino e teste) das pastas.
    
    Returns:
        X_train: Imagens de treino
        X_test: Imagens de teste
        y_train: Labels de treino (categorical)
        y_test: Labels de teste (categorical)
        dataset_stats: Dicionário com estatísticas do dataset
    """
    print('=' * 60)
    print('CARREGANDO DATASET FER-2013')
    print('=' * 60)
    
    # Carregar imagens de treino
    print('\n[1/2] Carregando imagens de TREINO...')
    X_train, y_train_raw, train_stats = load_images_from_directory(TRAIN_PATH)
    
    # Carregar imagens de teste
    print('\n[2/2] Carregando imagens de TESTE...')
    X_test, y_test_raw, test_stats = load_images_from_directory(TEST_PATH)
    
    # Converter labels para formato categórico (one-hot encoding)
    y_train = to_categorical(y_train_raw, num_classes=7)
    y_test = to_categorical(y_test_raw, num_classes=7)
    
    # Preparar estatísticas do dataset
    dataset_stats = {
        'train': {
            'total_images': len(X_train),
            'shape': X_train.shape,
            'per_emotion': train_stats,
            'distribution': {
                emotion: int(np.sum(y_train_raw == EMOTION_MAP[emotion]))
                for emotion in EMOTION_NAMES
            }
        },
        'test': {
            'total_images': len(X_test),
            'shape': X_test.shape,
            'per_emotion': test_stats,
            'distribution': {
                emotion: int(np.sum(y_test_raw == EMOTION_MAP[emotion]))
                for emotion in EMOTION_NAMES
            }
        },
        'validation_split': VALIDATION_SPLIT,
        'validation_size': int(len(X_train) * VALIDATION_SPLIT)
    }
    
    print('\n' + '=' * 60)
    print('DATASET CARREGADO COM SUCESSO!')
    print('=' * 60)
    print(f'Treino - Imagens: {X_train.shape}, Labels: {y_train.shape}')
    print(f'Teste  - Imagens: {X_test.shape}, Labels: {y_test.shape}')
    print(f'Validação (10% do treino): ~{dataset_stats["validation_size"]} imagens')
    print('=' * 60 + '\n')
    
    return X_train, X_test, y_train, y_test, dataset_stats


def create_model():
    """
    Cria o modelo de CNN para reconhecimento de emoções faciais.
    
    Returns:
        model: Modelo Keras compilado
        model_config: Dicionário com configuração do modelo
    """
    model = Sequential([
        # Primeira camada convolucional
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Segunda camada convolucional
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Camadas totalmente conectadas
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Regularização para evitar overfitting
        Dense(7, activation='softmax')  # 7 classes de emoções
    ])
    
    # Compilar o modelo
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Configuração do modelo para documentação
    model_config = {
        'architecture': 'CNN (Convolutional Neural Network)',
        'input_shape': (48, 48, 1),
        'layers': [
            {'type': 'Conv2D', 'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
            {'type': 'MaxPooling2D', 'pool_size': (2, 2)},
            {'type': 'Conv2D', 'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
            {'type': 'MaxPooling2D', 'pool_size': (2, 2)},
            {'type': 'Flatten'},
            {'type': 'Dense', 'units': 128, 'activation': 'relu'},
            {'type': 'Dropout', 'rate': 0.5},
            {'type': 'Dense', 'units': 7, 'activation': 'softmax'}
        ],
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy',
        'total_params': model.count_params()
    }
    
    return model, model_config


def calculate_metrics(model, X_test, y_test):
    """
    Calcula métricas detalhadas do modelo.
    
    Args:
        model: Modelo Keras treinado
        X_test: Dados de teste
        y_test: Labels de teste (categorical)
        
    Returns:
        metrics: Dicionário com todas as métricas
    """
    # Fazer predições
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calcular métricas gerais
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Calcular precision, recall, F1 para cada classe
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Calcular médias
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
    # Relatório de classificação
    report = classification_report(
        y_true, y_pred,
        target_names=EMOTION_NAMES,
        output_dict=True,
        zero_division=0
    )
    
    metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'per_class': {
            emotion: {
                'precision': float(precision[EMOTION_MAP[emotion]]),
                'recall': float(recall[EMOTION_MAP[emotion]]),
                'f1': float(f1[EMOTION_MAP[emotion]]),
                'support': int(support[EMOTION_MAP[emotion]])
            }
            for emotion in EMOTION_NAMES
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    return metrics


def visualize_training_history(history, save_path):
    """
    Visualiza o histórico de treinamento (accuracy e loss) e salva os gráficos.
    
    Args:
        history: Objeto History retornado pelo modelo.fit()
        save_path: Caminho onde salvar os gráficos
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico de Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', marker='o', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Gráfico de Loss
    ax2.plot(history.history['loss'], label='Train Loss', marker='o', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar gráfico combinado
    combined_path = os.path.join(save_path, 'training_history.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f'\nGráfico salvo em: {combined_path}')
    
    # Salvar gráficos individuais
    # Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    accuracy_path = os.path.join(save_path, 'accuracy.png')
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss', marker='o', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='s', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    loss_path = os.path.join(save_path, 'loss.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Gráficos individuais salvos em: {save_path}')
    plt.show()


def generate_markdown_report(test_id, start_time, end_time, dataset_stats, model_config, 
                            training_params, history, metrics, test_dir):
    """
    Gera um relatório Markdown automático com todas as informações do teste.
    
    Args:
        test_id: Identificador único do teste
        start_time: Data/hora de início
        end_time: Data/hora de término
        dataset_stats: Estatísticas do dataset
        model_config: Configuração do modelo
        training_params: Parâmetros de treinamento
        history: Histórico de treinamento
        metrics: Métricas calculadas
        test_dir: Diretório onde salvar o relatório
        
    Returns:
        report_path: Caminho do arquivo Markdown gerado
    """
    docs_dir = os.path.join(test_dir, 'docs')
    report_path = os.path.join(docs_dir, 'test_report.md')
    
    # Calcular duração do treinamento
    duration = end_time - start_time
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Analisar histórico para detectar problemas
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    # Detectar overfitting/underfitting
    observations = []
    if final_train_acc > final_val_acc + 0.1:
        observations.append("⚠️ **Possível overfitting detectado**: Diferença significativa entre acurácia de treino e validação.")
    if final_val_loss > final_train_loss * 1.2:
        observations.append("⚠️ **Validação loss maior que treino**: Modelo pode estar se adaptando demais aos dados de treino.")
    if metrics['test_accuracy'] < TARGET_ACCURACY_MIN:
        observations.append(f"ℹ️ **Acurácia abaixo do objetivo**: Modelo alcançou {metrics['test_accuracy']*100:.2f}%, objetivo é 75-89%.")
    elif metrics['test_accuracy'] >= TARGET_ACCURACY_MIN and metrics['test_accuracy'] <= TARGET_ACCURACY_MAX:
        observations.append(f"✅ **Objetivo alcançado**: Acurácia de {metrics['test_accuracy']*100:.2f}% está dentro do objetivo (75-89%).")
    else:
        observations.append(f"🎉 **Acurácia acima do objetivo**: Modelo superou a meta com {metrics['test_accuracy']*100:.2f}%!")
    
    if not observations:
        observations.append("✓ Treinamento concluído sem problemas aparentes.")
    
    # Gerar conteúdo Markdown
    md_content = f"""# Relatório de Teste - Reconhecimento de Emoções Faciais

## 📋 Identificador do Teste

- **ID do Teste**: `{test_id}`
- **Data/Hora de Início**: {start_time.strftime('%d/%m/%Y %H:%M:%S')}
- **Data/Hora de Término**: {end_time.strftime('%d/%m/%Y %H:%M:%S')}
- **Duração do Treinamento**: {int(hours)}h {int(minutes)}min {int(seconds)}s
- **Random Seed**: {RANDOM_SEED}

---

## 🎯 Objetivo de Desempenho

**Objetivo de acurácia: entre 75% e 89%.**

Acurácia obtida neste teste: **{metrics['test_accuracy']*100:.2f}%**

"""

    # Status do objetivo
    if metrics['test_accuracy'] < TARGET_ACCURACY_MIN:
        md_content += f"⚠️ **Status**: Acurácia abaixo do objetivo mínimo ({TARGET_ACCURACY_MIN*100}%)\n\n"
    elif metrics['test_accuracy'] > TARGET_ACCURACY_MAX:
        md_content += f"🎉 **Status**: Acurácia acima do objetivo máximo ({TARGET_ACCURACY_MAX*100}%)\n\n"
    else:
        md_content += f"✅ **Status**: Objetivo alcançado! Acurácia dentro da faixa desejada.\n\n"

    md_content += f"""---

## 📊 Parâmetros de Treinamento

### Hiperparâmetros
- **Épocas (Epochs)**: {training_params['epochs']}
- **Batch Size**: {training_params['batch_size']}
- **Validation Split**: {training_params['validation_split']*100}%
- **Optimizer**: {training_params['optimizer']}
- **Loss Function**: {training_params['loss']}
- **Random Seed**: {RANDOM_SEED}

### Arquitetura do Modelo
- **Tipo**: {model_config['architecture']}
- **Input Shape**: {model_config['input_shape']}
- **Total de Parâmetros**: {model_config['total_params']:,}

#### Estrutura das Camadas:
"""
    
    for i, layer in enumerate(model_config['layers'], 1):
        layer_type = layer['type']
        if layer_type == 'Conv2D':
            md_content += f"{i}. **{layer_type}**: {layer['filters']} filtros, kernel {layer['kernel_size']}, ativação {layer['activation']}\n"
        elif layer_type == 'MaxPooling2D':
            md_content += f"{i}. **{layer_type}**: Pool size {layer['pool_size']}\n"
        elif layer_type == 'Dense':
            md_content += f"{i}. **{layer_type}**: {layer['units']} neurônios, ativação {layer['activation']}\n"
        elif layer_type == 'Dropout':
            md_content += f"{i}. **{layer_type}**: Taxa {layer['rate']*100}%\n"
        else:
            md_content += f"{i}. **{layer_type}**\n"
    
    md_content += f"""

---

## 📁 Informações sobre o Dataset

### Dataset Utilizado
- **Nome**: FER-2013 (Facial Expression Recognition 2013)
- **Formato**: Imagens JPG organizadas por pastas de emoções
- **Resolução**: 48x48 pixels (escala de cinza)

### Divisão dos Dados
- **Treino**: {dataset_stats['train']['total_images']:,} imagens
  - Validação (10% do treino): ~{dataset_stats['validation_size']:,} imagens
  - Treino efetivo: ~{dataset_stats['train']['total_images'] - dataset_stats['validation_size']:,} imagens
- **Teste**: {dataset_stats['test']['total_images']:,} imagens

### Distribuição por Emoção - Treino
"""
    
    for emotion in EMOTION_NAMES:
        count = dataset_stats['train']['distribution'][emotion]
        percentage = (count / dataset_stats['train']['total_images']) * 100
        md_content += f"- **{emotion.capitalize()}**: {count:,} imagens ({percentage:.1f}%)\n"
    
    md_content += "\n### Distribuição por Emoção - Teste\n"
    
    for emotion in EMOTION_NAMES:
        count = dataset_stats['test']['distribution'][emotion]
        percentage = (count / dataset_stats['test']['total_images']) * 100
        md_content += f"- **{emotion.capitalize()}**: {count:,} imagens ({percentage:.1f}%)\n"
    
    md_content += f"""

---

## 📈 Métricas Principais

### Métricas Gerais
- **Acurácia (Accuracy)**: {metrics['test_accuracy']*100:.2f}%
- **Loss**: {metrics['test_loss']:.4f}
- **Precision (Macro)**: {metrics['precision_macro']*100:.2f}%
- **Recall (Macro)**: {metrics['recall_macro']*100:.2f}%
- **F1-Score (Macro)**: {metrics['f1_macro']*100:.2f}%
- **Precision (Weighted)**: {metrics['precision_weighted']*100:.2f}%
- **Recall (Weighted)**: {metrics['recall_weighted']*100:.2f}%
- **F1-Score (Weighted)**: {metrics['f1_weighted']*100:.2f}%

### Métricas por Classe
"""
    
    for emotion in EMOTION_NAMES:
        per_class = metrics['per_class'][emotion]
        md_content += f"""
#### {emotion.capitalize()}
- **Precision**: {per_class['precision']*100:.2f}%
- **Recall**: {per_class['recall']*100:.2f}%
- **F1-Score**: {per_class['f1']*100:.2f}%
- **Support**: {per_class['support']} amostras
"""
    
    md_content += "\n### Evolução durante o Treinamento\n"
    md_content += f"- **Acurácia Final (Treino)**: {final_train_acc*100:.2f}%\n"
    md_content += f"- **Acurácia Final (Validação)**: {final_val_acc*100:.2f}%\n"
    md_content += f"- **Loss Final (Treino)**: {final_train_loss:.4f}\n"
    md_content += f"- **Loss Final (Validação)**: {final_val_loss:.4f}\n"
    
    md_content += f"""

---

## 📸 Gráficos e Visualizações

Os seguintes gráficos foram gerados e salvos nesta pasta:

1. **training_history.png**: Gráfico combinado de accuracy e loss
2. **accuracy.png**: Gráfico detalhado de accuracy (treino e validação)
3. **loss.png**: Gráfico detalhado de loss (treino e validação)

### Localização dos Arquivos:
- `{os.path.relpath(test_dir)}/training_history.png`
- `{os.path.relpath(test_dir)}/accuracy.png`
- `{os.path.relpath(test_dir)}/loss.png`

---

## 📝 Observações e Logs

### Análise do Treinamento
"""
    
    for obs in observations:
        md_content += f"- {obs}\n"
    
    md_content += f"""

### Histórico de Épocas
"""
    
    # Adicionar tabela com histórico das últimas 5 épocas
    md_content += "\n#### Últimas 5 Épocas:\n\n"
    md_content += "| Época | Train Acc | Val Acc | Train Loss | Val Loss |\n"
    md_content += "|-------|-----------|---------|------------|----------|\n"
    
    for i in range(max(0, len(history.history['accuracy'])-5), len(history.history['accuracy'])):
        epoch = i + 1
        train_acc = history.history['accuracy'][i]
        val_acc = history.history['val_accuracy'][i]
        train_loss = history.history['loss'][i]
        val_loss = history.history['val_loss'][i]
        md_content += f"| {epoch} | {train_acc*100:.2f}% | {val_acc*100:.2f}% | {train_loss:.4f} | {val_loss:.4f} |\n"
    
    md_content += f"""
 

---

## 💾 Arquivos do Teste

- **Modelo Salvo**: `model_emotion_recognition.keras` (raiz do projeto)
- **Relatório**: Este arquivo (`test_report.md`)
- **Gráficos**: Pasta raiz deste teste

---

**Relatório gerado automaticamente em**: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
"""
    
    # Salvar arquivo Markdown
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f'\nRelatório Markdown gerado em: {report_path}')
    
    return report_path


# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================

if __name__ == '__main__':
    # Criar pasta do teste
    test_dir, docs_dir = create_test_directory()
    test_id = os.path.basename(test_dir)
    
    # Registrar início do treinamento
    start_time = datetime.now()
    print(f'\n{"="*60}')
    print(f'TREINAMENTO INICIADO - {start_time.strftime("%d/%m/%Y %H:%M:%S")}')
    print(f'{"="*60}\n')
    
    # Carregar dataset
    X_train, X_test, y_train, y_test, dataset_stats = load_dataset()
    
    # Criar modelo
    print('Criando modelo CNN...')
    model, model_config = create_model()
    
    print('\n' + '=' * 60)
    print('RESUMO DO MODELO')
    print('=' * 60)
    model.summary()
    print('=' * 60 + '\n')
    
    # Preparar parâmetros de treinamento
    training_params = {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'validation_split': VALIDATION_SPLIT,
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy',
        'random_seed': RANDOM_SEED
    }
    
    # Treinar modelo
    print('Iniciando treinamento...')
    print('=' * 60)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1
    )
    print('=' * 60)
    print('Treinamento concluído!')
    
    # Registrar término do treinamento
    end_time = datetime.now()
    
    # Avaliar modelo e calcular métricas
    print('\nAvaliando modelo no conjunto de teste...')
    metrics = calculate_metrics(model, X_test, y_test)
    
    print(f'\n{"="*60}')
    print('RESULTADOS DO TESTE')
    print('=' * 60)
    print(f'Acurácia: {metrics["test_accuracy"]*100:.2f}%')
    print(f'Loss: {metrics["test_loss"]:.4f}')
    print(f'Precision (Macro): {metrics["precision_macro"]*100:.2f}%')
    print(f'Recall (Macro): {metrics["recall_macro"]*100:.2f}%')
    print(f'F1-Score (Macro): {metrics["f1_macro"]*100:.2f}%')
    print('=' * 60)
    
    # Visualizar e salvar gráficos
    print('\nGerando gráficos...')
    visualize_training_history(history, test_dir)
    
    # Salvar modelo
    model_save_path = os.path.join(test_dir, 'model_emotion_recognition.keras')
    model.save(model_save_path)
    print(f'\nModelo salvo em: {model_save_path}')
    
    # Gerar relatório Markdown
    print('\nGerando relatório Markdown...')
    report_path = generate_markdown_report(
        test_id=test_id,
        start_time=start_time,
        end_time=end_time,
        dataset_stats=dataset_stats,
        model_config=model_config,
        training_params=training_params,
        history=history,
        metrics=metrics,
        test_dir=test_dir
    )
    
    print(f'\n{"="*60}')
    print(f'TREINAMENTO FINALIZADO - {end_time.strftime("%d/%m/%Y %H:%M:%S")}')
    print(f'{"="*60}')
    print(f'Pasta do teste: {test_dir}')
    print(f'Relatório: {report_path}')
    print('=' * 60 + '\n')
