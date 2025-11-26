"""
Módulo de utilidades para inferência de emoções em tempo real.
Contém funções para detecção facial, pré-processamento e predição.
"""

import os
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from tensorflow.keras.models import load_model


# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

EMOTION_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_LABELS_PT = {
    'angry': 'Raiva',
    'disgust': 'Nojo', 
    'fear': 'Medo',
    'happy': 'Feliz',
    'sad': 'Triste',
    'surprise': 'Surpresa',
    'neutral': 'Neutro'
}

# Cores para cada emoção (BGR)
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Vermelho
    'disgust': (0, 128, 0),    # Verde escuro
    'fear': (128, 0, 128),     # Roxo
    'happy': (0, 255, 255),    # Amarelo
    'sad': (255, 0, 0),        # Azul
    'surprise': (0, 165, 255), # Laranja
    'neutral': (128, 128, 128) # Cinza
}


# ============================================================================
# CLASSE PRINCIPAL DE INFERÊNCIA
# ============================================================================

class EmotionRecognizer:
    """
    Classe para reconhecimento de emoções em tempo real.
    """
    
    def __init__(self, model_path, use_dnn=False, smoothing_window=5):
        """
        Inicializa o reconhecedor de emoções.
        
        Args:
            model_path: Caminho para o modelo .keras treinado
            use_dnn: Se True, usa DNN Face Detector; se False, usa Haar Cascade
            smoothing_window: Tamanho da janela para suavização de predições
        """
        self.model = load_model(model_path)
        self.use_dnn = use_dnn
        self.smoothing_window = smoothing_window
        self.prediction_history = deque(maxlen=smoothing_window)
        
        # Inicializar detector facial
        self._init_face_detector()
        
    def _init_face_detector(self):
        """Inicializa o detector facial (Haar Cascade ou DNN)."""
        if self.use_dnn:
            # DNN Face Detector (mais preciso)
            proto_path = cv2.data.haarcascades + '../../../dnn/deploy.prototxt'
            model_path = cv2.data.haarcascades + '../../../dnn/res10_300x300_ssd_iter_140000.caffemodel'
            
            # Tentar caminhos alternativos para DNN
            if not os.path.exists(proto_path):
                # Usar arquivos locais se existirem
                proto_path = 'models/deploy.prototxt'
                model_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
                
            if os.path.exists(proto_path) and os.path.exists(model_path):
                self.face_detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                self.dnn_available = True
            else:
                print("⚠️ DNN não disponível, usando Haar Cascade como fallback")
                self.use_dnn = False
                self.dnn_available = False
                self._init_haar_cascade()
        else:
            self._init_haar_cascade()
            
    def _init_haar_cascade(self):
        """Inicializa o Haar Cascade para detecção facial."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
    def detect_faces_haar(self, frame):
        """
        Detecta rostos usando Haar Cascade.
        
        Args:
            frame: Frame BGR do OpenCV
            
        Returns:
            Lista de tuplas (x, y, w, h) com as coordenadas dos rostos
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )
        return faces
    
    def detect_faces_dnn(self, frame, confidence_threshold=0.5):
        """
        Detecta rostos usando DNN Face Detector.
        
        Args:
            frame: Frame BGR do OpenCV
            confidence_threshold: Limiar de confiança para detecção
            
        Returns:
            Lista de tuplas (x, y, w, h) com as coordenadas dos rostos
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2 - x1, y2 - y1))
                
        return faces
    
    def detect_faces(self, frame):
        """
        Detecta rostos no frame usando o método configurado.
        
        Args:
            frame: Frame BGR do OpenCV
            
        Returns:
            Lista de tuplas (x, y, w, h) com as coordenadas dos rostos
        """
        if self.use_dnn and hasattr(self, 'dnn_available') and self.dnn_available:
            return self.detect_faces_dnn(frame)
        return self.detect_faces_haar(frame)
    
    def preprocess_face(self, frame, face_coords):
        """
        Pré-processa o rosto para inferência (48x48, grayscale, normalizado).
        
        Args:
            frame: Frame BGR do OpenCV
            face_coords: Tupla (x, y, w, h) com coordenadas do rosto
            
        Returns:
            Array numpy pronto para inferência (1, 48, 48, 1)
        """
        x, y, w, h = face_coords
        
        # Garantir coordenadas válidas
        x = max(0, x)
        y = max(0, y)
        
        # Extrair região do rosto
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return None
            
        # Converter para escala de cinza
        if len(face_roi.shape) == 3:
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_roi
            
        # Redimensionar para 48x48
        face_resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_LANCZOS4)
        
        # Normalizar para [0, 1]
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # Adicionar dimensões para batch e canal
        face_input = np.expand_dims(face_normalized, axis=-1)
        face_input = np.expand_dims(face_input, axis=0)
        
        return face_input
    
    def predict_emotion(self, face_input, apply_smoothing=True):
        """
        Prediz a emoção a partir do rosto pré-processado.
        
        Args:
            face_input: Array numpy (1, 48, 48, 1)
            apply_smoothing: Se True, aplica suavização com janela deslizante
            
        Returns:
            emotion: Nome da emoção predita
            confidence: Confiança da predição (0-1)
            all_probs: Array com probabilidades de todas as emoções
        """
        # Fazer predição
        predictions = self.model.predict(face_input, verbose=0)[0]
        
        if apply_smoothing:
            # Adicionar ao histórico
            self.prediction_history.append(predictions)
            
            # Calcular média das predições
            if len(self.prediction_history) > 0:
                avg_predictions = np.mean(self.prediction_history, axis=0)
            else:
                avg_predictions = predictions
        else:
            avg_predictions = predictions
            
        # Obter emoção com maior probabilidade
        emotion_idx = np.argmax(avg_predictions)
        emotion = EMOTION_NAMES[emotion_idx]
        confidence = avg_predictions[emotion_idx]
        
        return emotion, confidence, avg_predictions
    
    def draw_results(self, frame, face_coords, emotion, confidence, fps=None):
        """
        Desenha bounding box, emoção e confiança no frame.
        
        Args:
            frame: Frame BGR do OpenCV
            face_coords: Tupla (x, y, w, h)
            emotion: Nome da emoção
            confidence: Confiança da predição
            fps: FPS atual (opcional)
            
        Returns:
            Frame com anotações
        """
        x, y, w, h = face_coords
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))
        emotion_pt = EMOTION_LABELS_PT.get(emotion, emotion)
        
        # Desenhar bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Preparar texto
        label = f"{emotion_pt}: {confidence*100:.1f}%"
        
        # Fundo do texto
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), color, -1)
        
        # Texto da emoção
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS no canto superior esquerdo
        if fps is not None:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        return frame
    
    def reset_smoothing(self):
        """Reseta o histórico de suavização."""
        self.prediction_history.clear()


# ============================================================================
# FUNÇÕES DE LOG E RELATÓRIO
# ============================================================================

def create_session_directory():
    """
    Cria uma nova pasta de sessão baseada em timestamp.
    
    Returns:
        session_dir: Caminho da pasta da sessão
        timestamp: String do timestamp
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join('Output', f'webcam_session_{timestamp}')
    snapshots_dir = os.path.join(session_dir, 'snapshots')
    
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(snapshots_dir, exist_ok=True)
    
    return session_dir, timestamp


def generate_session_report(session_dir, session_data):
    """
    Gera relatório Markdown da sessão de webcam.
    
    Args:
        session_dir: Diretório da sessão
        session_data: Dicionário com dados da sessão
    """
    report_path = os.path.join(session_dir, 'session_report.md')
    
    # Calcular estatísticas
    emotions_count = session_data.get('emotions_count', {})
    total_frames = session_data.get('total_frames', 0)
    
    if total_frames > 0:
        emotion_percentages = {
            e: (count / total_frames) * 100 
            for e, count in emotions_count.items()
        }
        predominant = max(emotions_count, key=emotions_count.get) if emotions_count else 'N/A'
    else:
        emotion_percentages = {}
        predominant = 'N/A'
    
    # Gerar conteúdo
    md_content = f"""# 📹 Relatório de Sessão - Reconhecimento de Emoções via Webcam

## 📋 Informações da Sessão

- **ID da Sessão**: `webcam_session_{session_data.get('timestamp', 'N/A')}`
- **Data/Hora de Início**: {session_data.get('start_time', 'N/A')}
- **Data/Hora de Término**: {session_data.get('end_time', 'N/A')}
- **Duração**: {session_data.get('duration', 'N/A')}
- **Modelo Utilizado**: `{session_data.get('model_path', 'N/A')}`
- **Detector Facial**: {session_data.get('detector_type', 'Haar Cascade')}

---

## 📊 Estatísticas da Sessão

### Métricas Gerais
- **Total de Frames Analisados**: {total_frames:,}
- **FPS Médio**: {session_data.get('avg_fps', 0):.1f}
- **Emoção Predominante**: **{EMOTION_LABELS_PT.get(predominant, predominant)}**

### Distribuição das Emoções

| Emoção | Contagem | Percentual |
|--------|----------|------------|
"""
    
    for emotion in EMOTION_NAMES:
        count = emotions_count.get(emotion, 0)
        pct = emotion_percentages.get(emotion, 0)
        emotion_pt = EMOTION_LABELS_PT.get(emotion, emotion)
        md_content += f"| {emotion_pt} | {count:,} | {pct:.1f}% |\n"
    
    md_content += f"""

---

## 📁 Arquivos Gerados

- **Relatório**: `session_report.md`
- **Log CSV**: `emotions_log.csv`
- **Snapshots**: `snapshots/` ({session_data.get('snapshots_count', 0)} imagens)

---

## 📝 Observações

{session_data.get('observations', '- Sessão concluída com sucesso.')}

---

**Relatório gerado automaticamente em**: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"\n📄 Relatório salvo em: {report_path}")
    return report_path

