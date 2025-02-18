import os
import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from collections import Counter, defaultdict
from mtcnn import MTCNN
import tensorflow as tf
import torch
from transformers import pipeline
from dotenv import load_dotenv

# Carrega variáveis do .env
load_dotenv()

# Configurações de cores e emojis
emotion_colors = {
    "neutral": (128, 128, 128),  # Cinza
    "happy": (0, 255, 0),        # Verde
    "sad": (255, 0, 0),          # Azul
    "angry": (0, 0, 255),        # Vermelho
    "surprise": (255, 255, 0),   # Amarelo
    "fear": (255, 0, 255),       # Magenta
    "disgust": (0, 255, 255)     # Ciano
}

# Restante das configurações e inicializações do código original...
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mtcnn_detector = MTCNN()

# Configuração da GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Dicionários de emojis (mantidos do código original)
emotion_emojis = {
    "neutral": "😐", "fear": "😨", "sad": "😢",
    "angry": "😠", "happy": "😊", "surprise": "😲",
    "disgust": "🤢"
}

activity_emojis = {
    "left_hand_raised": "✋ (Esq)", "right_hand_raised": "✋ (Dir)",
    "both_hands_raised": "🙌", "dancing": "💃",
    "handshake": "🤝", "standing": "🧍", "sitting": "🧘",
    "falling": "⚠️", "running": "🏃", "unusual_movement": "⚠️"
}

# Funções anteriores do código original (detect_faces, detect_dancing, etc.) permanecem

def detect_faces_with_emotions(frame):
    """
    Detecta rostos no frame com informações emocionais detalhadas.
    
    Args:
        frame (numpy.ndarray): Frame do vídeo para análise
    
    Returns:
        list: Lista de dicionários com informações de rostos e emoções
    """
    faces = []
    try:
        # Usar MTCNN para detecção inicial
        mtcnn_faces = mtcnn_detector.detect_faces(frame)
        
        for face in mtcnn_faces:
            if face['confidence'] > 0.90:
                x, y, w, h = face['box']
                face_frame = frame[y:y+h, x:x+w]
                
                if face_frame.size > 0:
                    try:
                        # Análise de emoção com DeepFace
                        result = DeepFace.analyze(
                            face_frame,
                            actions=['emotion'],
                            enforce_detection=False
                        )
                        
                        emotion_result = result[0]
                        emotion = emotion_result['dominant_emotion']
                        confidence = emotion_result['emotion'][emotion]
                        
                        faces.append({
                            'box': [x, y, w, h],
                            'emotion': emotion,
                            'confidence': confidence
                        })
                    except Exception as e:
                        print(f"Erro na análise emocional: {e}")
        
        return faces
    except Exception as e:
        print(f"Erro na detecção de rostos: {e}")
        return faces

def detect_anomalies(landmarks, previous_poses):
    anomalies = []
    
    if not isinstance(landmarks, np.ndarray):
        current_pose = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    else:
        current_pose = landmarks.reshape(-1, 3)
    
    nose = current_pose[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = current_pose[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = current_pose[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = current_pose[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = current_pose[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_ankle = current_pose[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = current_pose[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    if len(previous_poses) >= 2:
        prev_pose = previous_poses[-1].reshape(-1, 3)
        
        prev_hip_y = prev_pose[mp_pose.PoseLandmark.LEFT_HIP.value][1]
        hip_velocity = abs(left_hip[1] - prev_hip_y)
        hip_height = (left_hip[1] + right_hip[1]) / 2
        shoulder_height = (left_shoulder[1] + right_shoulder[1]) / 2
        
        if (hip_height > 0.8 and hip_velocity > 0.15 and 
            abs(shoulder_height - hip_height) < 0.2):
            anomalies.append("queda_detectada")

        velocities = np.linalg.norm(current_pose - prev_pose, axis=1)
        if np.max(velocities) > 0.25 and np.mean(velocities) > 0.15:
            anomalies.append("movimento_brusco")

    spine_vector = nose[:2] - (left_hip[:2] + right_hip[:2])/2
    vertical = np.array([0, -1])
    cos_angle = np.dot(spine_vector, vertical) / (np.linalg.norm(spine_vector) * np.linalg.norm(vertical))
    spine_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    if spine_angle > 45:
        anomalies.append("postura_anormal")

    ankle_distance = np.linalg.norm(left_ankle[:2] - right_ankle[:2])
    hip_distance = np.linalg.norm(left_hip[:2] - right_hip[:2])
    
    if ankle_distance > 2 * hip_distance:
        anomalies.append("desequilibrio")

    return list(set(anomalies))

def detect_activity(pose_landmarks, previous_poses):
    if not pose_landmarks:
        return "unknown"
    
    landmarks = pose_landmarks.landmark
    
    scores = {
        "standing": 0,
        "sitting": 0,
        "dancing": 0,
        "handshake": 0,
        "left_hand_raised": 0,
        "right_hand_raised": 0,
        "both_hands_raised": 0
    }
    
    left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
    
    if left_hand < left_shoulder - 0.2:
        scores["left_hand_raised"] += 1
    if right_hand < right_shoulder - 0.2:
        scores["right_hand_raised"] += 1
    if left_hand < left_shoulder - 0.2 and right_hand < right_shoulder - 0.2:
        scores["both_hands_raised"] += 2
    
    if detect_dancing(landmarks, previous_poses):
        scores["dancing"] += 2
    if detect_handshake(landmarks):
        scores["handshake"] += 2
    if detect_sitting(landmarks):
        scores["sitting"] += 2
    else:
        scores["standing"] += 1
    
    return max(scores.items(), key=lambda x: x[1])[0]
def detect_dancing(landmarks, previous_poses):
    if len(previous_poses) < 2:
        return False
        
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    
    current_pos = np.array([left_wrist.y, right_wrist.y])
    prev_pos = previous_poses[-1][[mp_pose.PoseLandmark.LEFT_WRIST.value, 
                                 mp_pose.PoseLandmark.RIGHT_WRIST.value], 1]
    
    velocity = np.mean(np.abs(current_pos - prev_pos))
    return velocity > 0.1

def detect_handshake(landmarks):
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    
    return (abs(left_wrist.x - right_wrist.x) < 0.15 and 
            abs(left_wrist.y - right_wrist.y) < 0.15)

def detect_sitting(landmarks):
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y
    
    return abs(hip - knee) < 0.3 and abs(knee - ankle) < 0.3

def main():
    cap = cv2.VideoCapture('video.mp4')
    if not cap.isOpened():
        print("Erro ao abrir o vídeo")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_video = cv2.VideoWriter(
        'advanced_video_analysis_with_emotions.mp4',
        cv2.VideoWriter_fourcc(*'avc1'),
        fps,
        (frame_width, frame_height)
    )
    
    # Restante da estrutura de dicionários do código original
    emotions_dict = defaultdict(list)
    activities_dict = defaultdict(list)
    timestamps_dict = defaultdict(list)
    anomalies_dict = defaultdict(list)
    previous_poses = []
    
    frame_skip = 1
    
    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1
    ) as pose:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Processamento de pose (código original)
                pose_results = pose.process(frame_rgb)
                if pose_results.pose_landmarks:
                    # Código de detecção de pose e atividades original
                    current_pose = np.array([[lm.x, lm.y, lm.z] 
                                           for lm in pose_results.pose_landmarks.landmark])
                    previous_poses.append(current_pose)
                    if len(previous_poses) > 5:
                        previous_poses.pop(0)
                    
                    # Detectar anomalias
                    anomalies = detect_anomalies(pose_results.pose_landmarks.landmark, previous_poses)
                    if anomalies:
                        anomalies_dict[frame_count].extend(anomalies)
                    
                    # Detectar atividade
                    activity = detect_activity(pose_results.pose_landmarks, previous_poses)
                    activities_dict[frame_count].append(activity)
                    
                    # Restante do processamento de pose e atividades original
                
                # Detecção de rostos com emoções
                faces = detect_faces_with_emotions(frame_rgb)
                for face in faces:
                    x, y, w, h = face['box']
                    emotion = face['emotion']
                    confidence = face['confidence']
                    
                    # Desenhar retângulo colorido
                    color = emotion_colors.get(emotion, (255, 255, 255))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Adicionar texto com emoji, emoção e confiança
                    emotion_text = f"{emotion_emojis.get(emotion, '😐')} {emotion}"
                    confidence_text = f"{confidence:.1f}%"
                    
                    # Fundo para texto
                    cv2.rectangle(frame, (x, y-40), (x+w, y), color, -1)
                    
                    # Emoção
                    cv2.putText(frame, emotion_text, (x+5, y-20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    
                    # Confiança
                    cv2.putText(frame, confidence_text, (x+5, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                
                output_video.write(frame)
                frame_count += 1
                
                cv2.imshow('Video Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
    
    # Manter função de geração de resumo original
    return generate_summary(emotions_dict, activities_dict, timestamps_dict, anomalies_dict, total_frames)

# Funções generate_summary e outras do código original permanecem inalteradas
def generate_summary(emotions_dict, activities_dict, timestamps_dict, anomalies_dict, total_frames):
    # Processamento de emoções com contagens absolutas e médias
    emotion_totals = defaultdict(int)
    weighted_emotions = defaultdict(float)
    emotion_counts = defaultdict(int)
    
    for person_emotions in emotions_dict.values():
        for emotion, confidence in person_emotions:
            emotion_totals[emotion] += 1
            weighted_emotions[emotion] += confidence
            emotion_counts[emotion] += 1
    
    # Calcular médias de confiança e porcentagens
    avg_emotions = {}
    emotion_percentages = {}
    for emotion in weighted_emotions:
        if emotion_counts[emotion] > 0:
            avg_emotions[emotion] = weighted_emotions[emotion] / emotion_counts[emotion]
            emotion_percentages[emotion] = (emotion_totals[emotion] / sum(emotion_totals.values())) * 100
    
    # Calcular atividades com contagens absolutas e porcentagens
    activity_counts = Counter([activity for activities in activities_dict.values() 
                             for activity in activities])
    total_activities = sum(activity_counts.values())
    activity_percentages = {
        activity: (count / total_activities) * 100
        for activity, count in activity_counts.items()
    }
    
    # Contagem de anomalias
    anomaly_counts = Counter([anomaly for anomalies in anomalies_dict.values() 
                            for anomaly in anomalies])
    
    return {
        "total_frames": total_frames,
        "emotions": {
            "counts": dict(emotion_totals),
            "confidence": avg_emotions,
            "percentages": emotion_percentages
        },
        "activities": {
            "counts": dict(activity_counts),
            "percentages": activity_percentages
        },
        "anomalies": anomaly_counts,
        "frames_without_emotions": (
            total_frames -
            sum(len(v) for v in emotions_dict.values())
        )
    }

if __name__ == "__main__":
    summary = main()
    
    print("\nResumo de Emoções, Atividades e Anomalias")
    print("=========================================")
    print(f"\nTotal de frames analisados: {summary['total_frames']}")
    
    print("\nEmoções detectadas:")
    total_emotions = sum(summary['emotions']['counts'].values())
    for emotion in summary['emotions']['counts']:
        count = summary['emotions']['counts'][emotion]
        confidence = summary['emotions']['confidence'][emotion]
        percentage = summary['emotions']['percentages'][emotion]
        print(f"{emotion_emojis.get(emotion, emotion)} {emotion}:")
        print(f"  - Ocorrências: {count} vezes")
        print(f"  - Porcentagem: {percentage:.2f}% do total de emoções detectadas")
        print(f"  - Confiança média: {confidence:.2f}%")
    
    print("\nAtividades detectadas:")
    for activity, count in summary['activities']['counts'].items():
        percentage = summary['activities']['percentages'][activity]
        print(f"{activity_emojis.get(activity, activity)} {activity}:")
        print(f"  - Ocorrências: {count} vezes")
        print(f"  - Porcentagem: {percentage:.2f}% do tempo")
    
    print("\nAnomalias detectadas:")
    for anomaly, count in summary['anomalies'].items():
        percentage = (count / summary['total_frames']) * 100
        print(f"⚠️ {anomaly}:")
        print(f"  - Ocorrências: {count} vezes")
        print(f"  - Porcentagem: {percentage:.2f}% dos frames")
    
    frames_without_emotions = summary['frames_without_emotions']
    emotion_absence_percentage = (frames_without_emotions/summary['total_frames'])*100
    print(f"\nFrames sem emoções detectadas:")
    print(f"  - Quantidade: {frames_without_emotions}")
    print(f"  - Porcentagem: {emotion_absence_percentage:.2f}% do total de frames")
