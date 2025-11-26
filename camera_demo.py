"""
Demo de Reconhecimento de Emoções em Tempo Real via Webcam.

Uso:
    python camera_demo.py
    python camera_demo.py --model-path Output/test_20251122_232348/model_emotion_recognition.keras
    python camera_demo.py --use-dnn --smoothing 7
    python camera_demo.py --save-snapshots --snapshot-interval 30
"""

import os
import cv2
import csv
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict

from inference_utils import (
    EmotionRecognizer,
    create_session_directory,
    generate_session_report,
    EMOTION_NAMES,
    EMOTION_LABELS_PT
)


def parse_arguments():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description='Reconhecimento de Emoções Faciais em Tempo Real',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
    python camera_demo.py
    python camera_demo.py --model-path Output/test_20251122_232348/model_emotion_recognition.keras
    python camera_demo.py --camera-index 1 --use-dnn
    python camera_demo.py --save-snapshots --snapshot-interval 60
    
Controles durante execução:
    Q ou ESC - Sair
    S        - Salvar snapshot manualmente
    R        - Resetar suavização
        """
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Caminho para o modelo .keras (padrão: busca o mais recente em Output/)'
    )
    
    parser.add_argument(
        '--camera-index',
        type=int,
        default=0,
        help='Índice da câmera (padrão: 0)'
    )
    
    parser.add_argument(
        '--use-dnn',
        action='store_true',
        help='Usar DNN Face Detector ao invés de Haar Cascade'
    )
    
    parser.add_argument(
        '--smoothing',
        type=int,
        default=5,
        help='Tamanho da janela de suavização (padrão: 5)'
    )
    
    parser.add_argument(
        '--save-snapshots',
        action='store_true',
        help='Salvar snapshots automaticamente'
    )
    
    parser.add_argument(
        '--snapshot-interval',
        type=int,
        default=30,
        help='Intervalo de frames entre snapshots automáticos (padrão: 30)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Executar sem exibir janela (apenas log)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Largura do frame da câmera (padrão: 640)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Altura do frame da câmera (padrão: 480)'
    )
    
    return parser.parse_args()


def find_latest_model():
    """Encontra o modelo mais recente na pasta Output."""
    output_dir = 'Output'
    if not os.path.exists(output_dir):
        return None
    
    # Listar pastas de teste
    test_dirs = [
        d for d in os.listdir(output_dir) 
        if d.startswith('test_') and os.path.isdir(os.path.join(output_dir, d))
    ]
    
    if not test_dirs:
        return None
    
    # Ordenar por nome (timestamp)
    test_dirs.sort(reverse=True)
    
    # Procurar modelo na pasta mais recente
    for test_dir in test_dirs:
        model_path = os.path.join(output_dir, test_dir, 'model_emotion_recognition.keras')
        if os.path.exists(model_path):
            return model_path
    
    return None


def main():
    """Função principal do demo de webcam."""
    args = parse_arguments()
    
    # =========================================================================
    # INICIALIZAÇÃO
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("🎥 RECONHECIMENTO DE EMOÇÕES EM TEMPO REAL")
    print("=" * 60)
    
    # Encontrar modelo
    model_path = args.model_path
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            print("❌ Erro: Nenhum modelo encontrado!")
            print("   Execute primeiro o treinamento (python main.py)")
            print("   Ou especifique o caminho: --model-path <caminho>")
            return
    
    if not os.path.exists(model_path):
        print(f"❌ Erro: Modelo não encontrado: {model_path}")
        return
    
    print(f"\n📦 Modelo: {model_path}")
    print(f"📷 Câmera: {args.camera_index}")
    print(f"🔍 Detector: {'DNN' if args.use_dnn else 'Haar Cascade'}")
    print(f"📊 Suavização: {args.smoothing} frames")
    
    # Criar diretório da sessão
    session_dir, timestamp = create_session_directory()
    print(f"📁 Sessão: {session_dir}")
    
    # Inicializar reconhecedor
    print("\n⏳ Carregando modelo...")
    recognizer = EmotionRecognizer(
        model_path=model_path,
        use_dnn=args.use_dnn,
        smoothing_window=args.smoothing
    )
    print("✅ Modelo carregado!")
    
    # Inicializar câmera
    print(f"\n⏳ Inicializando câmera {args.camera_index}...")
    cap = cv2.VideoCapture(args.camera_index)
    
    if not cap.isOpened():
        print(f"❌ Erro: Não foi possível abrir a câmera {args.camera_index}")
        return
    
    # Configurar resolução
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    print("✅ Câmera inicializada!")
    print("\n" + "-" * 60)
    print("Controles: Q/ESC=Sair | S=Snapshot | R=Reset suavização")
    print("-" * 60 + "\n")
    
    # =========================================================================
    # PREPARAR LOGS
    # =========================================================================
    
    csv_path = os.path.join(session_dir, 'emotions_log.csv')
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'frame', 'emotion', 'emotion_pt', 'confidence'] + 
                        [f'prob_{e}' for e in EMOTION_NAMES])
    
    # =========================================================================
    # VARIÁVEIS DE CONTROLE
    # =========================================================================
    
    start_time = datetime.now()
    frame_count = 0
    fps_list = []
    emotions_count = defaultdict(int)
    snapshots_count = 0
    prev_time = datetime.now()
    
    # =========================================================================
    # LOOP PRINCIPAL
    # =========================================================================
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Erro ao capturar frame")
                break
            
            frame_count += 1
            current_time = datetime.now()
            
            # Calcular FPS
            time_diff = (current_time - prev_time).total_seconds()
            if time_diff > 0:
                fps = 1.0 / time_diff
                fps_list.append(fps)
            else:
                fps = 0
            prev_time = current_time
            
            # Detectar rostos
            faces = recognizer.detect_faces(frame)
            
            # Processar cada rosto detectado
            for face_coords in faces:
                # Pré-processar
                face_input = recognizer.preprocess_face(frame, face_coords)
                
                if face_input is None:
                    continue
                
                # Predizer emoção
                emotion, confidence, all_probs = recognizer.predict_emotion(face_input)
                
                # Atualizar contagem
                emotions_count[emotion] += 1
                
                # Log CSV
                csv_writer.writerow([
                    current_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    frame_count,
                    emotion,
                    EMOTION_LABELS_PT.get(emotion, emotion),
                    f'{confidence:.4f}'
                ] + [f'{p:.4f}' for p in all_probs])
                
                # Desenhar resultados
                frame = recognizer.draw_results(frame, face_coords, emotion, confidence, fps)
            
            # Salvar snapshot automático
            if args.save_snapshots and frame_count % args.snapshot_interval == 0 and len(faces) > 0:
                snapshot_path = os.path.join(
                    session_dir, 'snapshots', 
                    f'snapshot_{frame_count:06d}.jpg'
                )
                cv2.imwrite(snapshot_path, frame)
                snapshots_count += 1
            
            # Exibir frame
            if not args.no_display:
                # Adicionar instruções na tela
                cv2.putText(frame, "Q/ESC: Sair | S: Snapshot | R: Reset", 
                           (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow('Reconhecimento de Emocoes', frame)
                
                # Processar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q ou ESC
                    print("\n🛑 Encerrando sessão...")
                    break
                elif key == ord('s'):  # Snapshot manual
                    snapshot_path = os.path.join(
                        session_dir, 'snapshots',
                        f'manual_{datetime.now().strftime("%H%M%S")}.jpg'
                    )
                    cv2.imwrite(snapshot_path, frame)
                    snapshots_count += 1
                    print(f"📸 Snapshot salvo: {snapshot_path}")
                elif key == ord('r'):  # Reset suavização
                    recognizer.reset_smoothing()
                    print("🔄 Suavização resetada")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrompido pelo usuário")
    
    finally:
        # =====================================================================
        # FINALIZAÇÃO
        # =====================================================================
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Fechar recursos
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        
        # Calcular estatísticas finais
        avg_fps = np.mean(fps_list) if fps_list else 0
        
        # Preparar dados da sessão
        session_data = {
            'timestamp': timestamp,
            'start_time': start_time.strftime('%d/%m/%Y %H:%M:%S'),
            'end_time': end_time.strftime('%d/%m/%Y %H:%M:%S'),
            'duration': str(duration).split('.')[0],
            'model_path': model_path,
            'detector_type': 'DNN Face Detector' if args.use_dnn else 'Haar Cascade',
            'total_frames': sum(emotions_count.values()),
            'avg_fps': avg_fps,
            'emotions_count': dict(emotions_count),
            'snapshots_count': snapshots_count,
            'observations': '- Sessão concluída com sucesso.'
        }
        
        # Gerar relatório
        generate_session_report(session_dir, session_data)
        
        # Resumo final
        print("\n" + "=" * 60)
        print("📊 RESUMO DA SESSÃO")
        print("=" * 60)
        print(f"⏱️  Duração: {session_data['duration']}")
        print(f"🖼️  Frames analisados: {session_data['total_frames']:,}")
        print(f"📈 FPS médio: {avg_fps:.1f}")
        print(f"📸 Snapshots salvos: {snapshots_count}")
        print(f"📁 Arquivos em: {session_dir}")
        
        if emotions_count:
            predominant = max(emotions_count, key=emotions_count.get)
            print(f"😊 Emoção predominante: {EMOTION_LABELS_PT.get(predominant, predominant)}")
        
        print("=" * 60 + "\n")


if __name__ == '__main__':
    main()

