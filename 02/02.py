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

# Recupera o token de autenticação da variável de ambiente
access_token = os.getenv("HF_TOKEN")
if access_token is None:
    raise ValueError("O token de acesso não foi encontrado. Defina a variável de ambiente 'HF_TOKEN' com seu token de acesso.")

# Inicialização dos modelos
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mtcnn_detector = MTCNN()

# Configuração da GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Dicionários de emojis
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

def detect_faces(frame):
    faces = []
    try:
        mtcnn_faces = mtcnn_detector.detect_faces(frame)
        
        for face in mtcnn_faces:
            if face['confidence'] > 0.95:
                faces.append({
                    'box': [int(face['box'][0]), 
                           int(face['box'][1]),
                           int(face['box'][2]), 
                           int(face['box'][3])],
                    'confidence': face['confidence']
                })
        
        return faces
    except Exception as e:
        print(f"Erro na detecção de rostos: {e}")
        return faces

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

def detect_anomalies(landmarks, previous_poses):
    anomalies = []
    
    # Converte landmarks para formato consistente
    if not isinstance(landmarks, np.ndarray):
        current_pose = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    else:
        current_pose = landmarks.reshape(-1, 3)
    
    # Extrai coordenadas principais
    nose = current_pose[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = current_pose[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = current_pose[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = current_pose[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = current_pose[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_ankle = current_pose[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = current_pose[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    if len(previous_poses) >= 2:
        prev_pose = previous_poses[-1].reshape(-1, 3)
        
        # Detecta queda
        prev_hip_y = prev_pose[mp_pose.PoseLandmark.LEFT_HIP.value][1]
        hip_velocity = abs(left_hip[1] - prev_hip_y)
        hip_height = (left_hip[1] + right_hip[1]) / 2
        shoulder_height = (left_shoulder[1] + right_shoulder[1]) / 2
        
        if (hip_height > 0.8 and 
            hip_velocity > 0.15 and 
            abs(shoulder_height - hip_height) < 0.2):
            anomalies.append("queda_detectada")

        # Detecta movimento brusco
        velocities = np.linalg.norm(current_pose - prev_pose, axis=1)
        if np.max(velocities) > 0.25 and np.mean(velocities) > 0.15:
            anomalies.append("movimento_brusco")

    # Detecta postura anormal
    spine_vector = nose[:2] - (left_hip[:2] + right_hip[:2])/2
    vertical = np.array([0, -1])
    cos_angle = np.dot(spine_vector, vertical) / (np.linalg.norm(spine_vector) * np.linalg.norm(vertical))
    spine_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    if spine_angle > 45:
        anomalies.append("postura_anormal")

    # Detecta desequilíbrio
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
        'advanced_video_analysis.mp4',
        cv2.VideoWriter_fourcc(*'avc1'),
        fps,
        (frame_width, frame_height)
    )
    
    emotions_dict = defaultdict(list)
    activities_dict = defaultdict(list)
    timestamps_dict = defaultdict(list)
    anomalies_dict = defaultdict(list)
    previous_poses = []
    
    frame_skip = 3
    
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
                
                pose_results = pose.process(frame_rgb)
                if pose_results.pose_landmarks:
                    current_pose = np.array([[lm.x, lm.y, lm.z] 
                                           for lm in pose_results.pose_landmarks.landmark])
                    previous_poses.append(current_pose)
                    if len(previous_poses) > 5:
                        previous_poses.pop(0)
                    
                    # Detectar anomalias
                    anomalies = detect_anomalies(pose_results.pose_landmarks.landmark, previous_poses)
                    if anomalies:
                        anomalies_dict[frame_count].extend(anomalies)
                        
                        cv2.putText(
                            frame,
                            f"ANOMALIA: {', '.join(anomalies)}",
                            (10, frame_height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )
                    
                    mp_drawing.draw_landmarks(
                        frame,
                        pose_results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )
                    
                    activity = detect_activity(pose_results.pose_landmarks, previous_poses)
                    activities_dict[frame_count].append(activity)
                    
                    text = f"Activity: {activity_emojis.get(activity, activity)}"
                    (text_width, text_height), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(
                        frame,
                        (10, 35),
                        (10 + text_width, 65),
                        (0, 0, 0),
                        -1
                    )
                    cv2.putText(
                        frame,
                        text,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                
                faces = detect_faces(frame_rgb)
                for i, face in enumerate(faces):
                    x, y, w, h = face['box']
                    face_frame = frame[y:y+h, x:x+w]
                    
                    if face_frame.size > 0:
                        try:
                            result = DeepFace.analyze(
                                face_frame,
                                actions=['emotion'],
                                enforce_detection=False,
                                detector_backend='opencv'
                            )
                            
                            if result:
                                emotion = result[0]['dominant_emotion']
                                confidence = result[0]['emotion'][emotion]
                                
                                if confidence > 40:
                                    emotions_dict[i].append((emotion, confidence))
                                    
                                    bg_color = (0, 0, 0)  # Default black
                                    if emotion in ['angry', 'fear']:
                                        bg_color = (0, 0, 255)  # Red
                                    elif emotion in ['happy', 'surprise']:
                                        bg_color = (0, 255, 0)  # Green
                                        
                                    text = f"{emotion_emojis.get(emotion, emotion)} {confidence:.1f}%"
                                    
                                    (text_width, text_height), _ = cv2.getTextSize(
                                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                    
                                    cv2.rectangle(
                                        frame,
                                        (x, y-35),
                                        (x + text_width, y-5),
                                        bg_color,
                                        -1
                                    )
                                    cv2.putText(
                                        frame,
                                        text,
                                        (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (255, 255, 255),
                                        2
                                    )
                                    
                                    timestamps_dict[i].append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
                        except Exception as e:
                            print(f"Erro na análise de emoção: {e}")
            
            # Adicionar barra de progresso
            progress = (frame_count / total_frames) * 100
            cv2.rectangle(frame, (10, frame_height - 30), 
                         (int(10 + (frame_width - 20) * progress / 100), frame_height - 20),
                         (0, 255, 0), -1)
            
            # Adicionar timestamp
            text = f"Time: {cap.get(cv2.CAP_PROP_POS_MSEC)/1000:.1f}s"
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(
                frame,
                (10, 5),
                (10 + text_width, 35),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            output_video.write(frame)
            frame_count += 1
            
            cv2.imshow('Video Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
    
    return generate_summary(emotions_dict, activities_dict, timestamps_dict, anomalies_dict, total_frames)

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
