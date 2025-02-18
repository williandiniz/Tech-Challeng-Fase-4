import cv2
import os
import numpy as np
from tqdm import tqdm
from retinaface import RetinaFace
from deepface import DeepFace
import mediapipe as mp
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import whisper
import time

def analyze_video(video_path, output_path):
    # Inicializar detectores
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Inicializar variáveis para análise
    emotions_data = []
    motion_data = []
    pose_data = []
    
    # Adicionar contadores para anomalias
    anomaly_count = 0
    analyzed_frames = 0
    
    # Definir thresholds para movimentos anômalos
    movement_thresholds = {
        'head': 0.3,  # Movimento brusco da cabeça
        'arms': 0.4,  # Movimento brusco dos braços
        'legs': 0.5,  # Movimento brusco das pernas
        'sudden_movement': 0.6  # Movimento súbito do corpo todo
    }
    
    # Capturar vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return
    
    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Configurar VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Variáveis para análise de movimento
    previous_landmarks = None
    movement_threshold = 0.1

    # Criar janela para exibição
    cv2.namedWindow('Processamento de Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Processamento de Video', 1280, 720)
    
    # Variáveis para timestamp
    start_time = datetime.now()
    
    for frame_count in tqdm(range(total_frames), desc="Processando vídeo"):
        analyzed_frames += 1
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = start_time + timedelta(seconds=frame_count/fps)
        
        # Converter BGR para RGB para processamento
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar pose
        pose_results = pose.process(frame_rgb)
        
        # Análise de movimento corporal
        if pose_results.pose_landmarks:
            # Desenhar landmarks do corpo
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            
            # Análise de movimentos específicos
            landmarks = pose_results.pose_landmarks.landmark
            
            # Detectar movimentos anômalos
            if previous_landmarks:
                head_movement = calculate_head_movement(landmarks, previous_landmarks)
                arm_movement = calculate_arm_movement(landmarks, previous_landmarks)
                leg_movement = calculate_leg_movement(landmarks, previous_landmarks)
                
                # Verificar movimentos que excedem os thresholds
                if (head_movement > movement_thresholds['head'] or
                    arm_movement > movement_thresholds['arms'] or
                    leg_movement > movement_thresholds['legs']):
                    anomaly_count += 1
                    
                    # Adicionar informação de anomalia ao frame
                    cv2.putText(frame, "MOVIMENTO ANÔMALO DETECTADO", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 0, 255), 2)
            
            # Detectar tipos de movimento
            movement_type = analyze_movement(landmarks, previous_landmarks)
            if movement_type:
                pose_data.append({
                    'timestamp': timestamp,
                    'movement_type': movement_type
                })
            
            previous_landmarks = landmarks
        
        # Detectar faces e emoções
        faces = RetinaFace.detect_faces(frame)
        
        if isinstance(faces, dict):
            for face_idx, face_data in faces.items():
                facial_area = face_data['facial_area']
                x1, y1, x2, y2 = facial_area
                
                face_image = frame[y1:y2, x1:x2]
                
                try:
                    emotion_analysis = DeepFace.analyze(face_image, 
                                                      actions=['emotion'],
                                                      enforce_detection=False)
                    
                    if isinstance(emotion_analysis, list):
                        emotion_analysis = emotion_analysis[0]
                    
                    dominant_emotion = emotion_analysis['dominant_emotion']
                    emotion_scores = emotion_analysis['emotion']
                    
                    emotions_data.append({
                        'timestamp': timestamp,
                        'face_id': face_idx,
                        'dominant_emotion': dominant_emotion,
                        'emotion_scores': emotion_scores
                    })
                    
                    # Desenhar retângulo e emoção
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{dominant_emotion}", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                except Exception as e:
                    print(f"Erro na análise de emoção: {str(e)}")
                    continue
        
        # Exibir frame processado
        cv2.imshow('Processamento de Video', frame)
        
        # Adicionar pequeno delay para visualização
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Salvar frame processado
        out.write(frame)
    
    # Liberar recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Transcrever áudio usando Whisper
    print("\nIniciando transcrição do áudio...")
    transcribe_audio_whisper(video_path, output_path.replace('.mp4', '_transcription.txt'))
    
    # Gerar relatório
    generate_report(emotions_data, motion_data, pose_data, analyzed_frames, anomaly_count, output_path.replace('.mp4', '_report.md'))
    
    return emotions_data, motion_data, pose_data, analyzed_frames, anomaly_count

def transcribe_audio_whisper(video_path, output_text_path):
    """Extrai e transcreve o áudio do vídeo para texto usando Whisper"""
    try:
        # Carregar modelo Whisper
        model = whisper.load_model("base")
        
        # Extrair áudio temporariamente
        video = VideoFileClip(video_path)
        temp_audio_path = tempfile.mktemp(suffix='.wav')
        video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
        video.close()
        
        # Transcrever usando Whisper
        result = model.transcribe(temp_audio_path, language="en")
        
        # Salvar transcrição
        with open(output_text_path, 'w', encoding='utf-8') as f:
            f.write(result["text"])
            
        # Remover arquivo temporário
        os.remove(temp_audio_path)
        print(f"Transcrição concluída e salva em: {output_text_path}")
        
    except Exception as e:
        print(f"Erro na transcrição do áudio: {str(e)}")

def analyze_movement(current_landmarks, previous_landmarks):
    """Analisa o tipo de movimento baseado nas landmarks"""
    if not previous_landmarks:
        return None
        
    movements = []
    
    # Calcular movimento da cabeça
    head_movement = calculate_head_movement(current_landmarks, previous_landmarks)
    if head_movement > 0.1:
        movements.append("Movimento de cabeça")
    
    # Calcular movimento dos braços
    arm_movement = calculate_arm_movement(current_landmarks, previous_landmarks)
    if arm_movement > 0.15:
        movements.append("Movimento de braços")
    
    # Calcular movimento das pernas
    leg_movement = calculate_leg_movement(current_landmarks, previous_landmarks)
    if leg_movement > 0.2:
        movements.append("Movimento de pernas")
    
    # Detectar caminhada
    if detect_walking(current_landmarks, previous_landmarks):
        movements.append("Caminhando")
    
    # Se não houver movimentos significativos
    if not movements:
        return "Parado"
    
    return ", ".join(movements)

def calculate_head_movement(curr, prev):
    """Calcula movimento da cabeça"""
    head_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return calculate_points_movement(curr, prev, head_points)

def calculate_arm_movement(curr, prev):
    """Calcula movimento dos braços"""
    arm_points = [11, 13, 15, 12, 14, 16]
    return calculate_points_movement(curr, prev, arm_points)

def calculate_leg_movement(curr, prev):
    """Calcula movimento das pernas"""
    leg_points = [23, 25, 27, 29, 31, 24, 26, 28, 30, 32]
    return calculate_points_movement(curr, prev, leg_points)

def calculate_points_movement(curr, prev, points):
    """Calcula movimento médio entre pontos específicos"""
    total_movement = 0
    for point in points:
        curr_point = np.array([curr[point].x, curr[point].y, curr[point].z])
        prev_point = np.array([prev[point].x, prev[point].y, prev[point].z])
        movement = np.linalg.norm(curr_point - prev_point)
        total_movement += movement
    return total_movement / len(points)

def detect_walking(curr, prev):
    """Detecta movimento de caminhada baseado no movimento das pernas"""
    left_ankle = curr[27]
    right_ankle = curr[28]
    prev_left_ankle = prev[27]
    prev_right_ankle = prev[28]
    
    ankle_movement = (abs(left_ankle.y - prev_left_ankle.y) + 
                     abs(right_ankle.y - prev_right_ankle.y)) / 2
    return ankle_movement > 0.1

def generate_report(emotions_data, motion_data, pose_data, analyzed_frames, anomaly_count, report_path):
    """Gera um relatório detalhado da análise"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Relatório de Análise de Vídeo\n\n")
        f.write(f"Data da análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Adicionar resumo geral
        f.write("## Resumo Geral\n\n")
        f.write(f"- Total de frames analisados: {analyzed_frames}\n")
        f.write(f"- Número de anomalias detectadas: {anomaly_count}\n\n")
        
        if emotions_data:
            f.write("## Análise de Emoções\n\n")
            all_emotions = [data['dominant_emotion'] for data in emotions_data]
            emotion_counts = pd.Series(all_emotions).value_counts()
            
            f.write("### Distribuição de Emoções\n\n")
            for emotion, count in emotion_counts.items():
                f.write(f"- {emotion}: {count} ocorrências\n")
            
            f.write("\n### Emoções por Tempo\n\n")
            for data in emotions_data[:10]:
                timestamp = data['timestamp'].strftime('%H:%M:%S')
                f.write(f"- {timestamp}: Face {data['face_id']} - {data['dominant_emotion']}\n")
        
        if pose_data:
            f.write("\n## Análise de Movimentos\n\n")
            all_movements = [data['movement_type'] for data in pose_data]
            movement_counts = pd.Series(all_movements).value_counts()
            
            f.write("### Tipos de Movimento Detectados\n\n")
            for movement, count in movement_counts.items():
                f.write(f"- {movement}: {count} ocorrências\n")

# Uso do script
if __name__ == "__main__":
    script_dir = os.getcwd()
    input_video_path = os.path.join(script_dir, 'video.mp4')
    output_video_path = os.path.join(script_dir, 'output_video_analysis1.mp4')
    
    emotions_data, motion_data, pose_data, analyzed_frames, anomaly_count = analyze_video(input_video_path, output_video_path)
