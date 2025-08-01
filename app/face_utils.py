import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from io import BytesIO

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, refine_landmarks=True)

def calculate_beauty_score(asym, eye_ratio, jaw_ratio, forehead_w, nose_len, lip_thickness):
    score = 0

    if asym < 0.01:
        score += 30

    if 0.34 <= eye_ratio <= 0.42:
        score += 20

    if 0.45 <= jaw_ratio <= 0.55:
        score += 15

    if 0.35 <= forehead_w <= 0.5:
        score += 10

    if 0.12 <= nose_len <= 0.18:
        score += 15

    if 0.025 <= lip_thickness <= 0.04:
        score += 10

    return score

def process_face_image(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    h, w, _ = img_np.shape
    rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    res = face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        return {"error": "얼굴을 인식하지 못했습니다."}

    lm = res.multi_face_landmarks[0]
    debug_img = img_np.copy()

    # 전체 랜드마크 (녹색)
    for landmark in lm.landmark:
        cx = int(landmark.x * w)
        cy = int(landmark.y * h)
        cv2.circle(debug_img, (cx, cy), 1, (0, 255, 0), -1)

    # 주요 특징점
    def pt(i): return np.array([lm.landmark[i].x * w, lm.landmark[i].y * h])

    # 주요 좌표들
    points = {
        "left_eye": (33, (255, 0, 0)),         # 파란색
        "right_eye": (263, (255, 0, 0)),
        "left_cheek": (234, (0, 165, 255)),    # 주황
        "right_cheek": (454, (0, 165, 255)),
        "forehead1": (10, (255, 255, 0)),      # 하늘
        "forehead2": (336, (255, 255, 0)),
        "nose_top": (1, (0, 0, 255)),          # 빨강
        "nose_tip": (4, (0, 0, 255)),
        "upper_lip": (13, (255, 0, 255)),      # 보라
        "lower_lip": (14, (255, 0, 255)),
        "chin": (152, (0, 255, 255))           # 노랑
    }

    coords = {}
    for name, (idx, color) in points.items():
        x = int(lm.landmark[idx].x * w)
        y = int(lm.landmark[idx].y * h)
        coords[name] = np.array([x, y])
        cv2.circle(debug_img, (x, y), 4, color, -1)
        cv2.putText(debug_img, name, (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 디버깅 이미지 저장
    cv2.imwrite("face_debug_output.jpg", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

    # 계산
    eye_dist = np.linalg.norm(coords["left_eye"] - coords["right_eye"])
    face_w = np.linalg.norm(coords["left_cheek"] - coords["right_cheek"])
    face_h = np.linalg.norm(((coords["forehead1"] + coords["forehead2"]) / 2) - coords["chin"])
    jaw_length = np.linalg.norm(coords["nose_tip"] - coords["chin"])
    eye_ratio = eye_dist / face_w
    jaw_ratio = jaw_length / face_h
    forehead_w = abs(coords["forehead1"][0] - coords["forehead2"][0]) / face_w
    nose_len = np.linalg.norm(coords["nose_top"] - coords["nose_tip"]) / face_h
    lip_thickness = abs(coords["upper_lip"][1] - coords["lower_lip"][1]) / face_h
    asym = abs(lm.landmark[10].x - (1 - lm.landmark[336].x))


    results = []
    if eye_ratio > 0.42:
        results.append("양쪽 눈 사이 간격이 넓어 타인과의 거리감을 덜 느끼며, 새로운 환경과 사람에게도 적극적으로 다가가는 외향적인 성격입니다. 활발하고 개방적인 사회성을 지니고 있어 리더나 분위기 메이커로 활약할 수 있습니다.")
    elif eye_ratio < 0.34:
        results.append("눈 간격이 좁은 편으로, 집중력이 뛰어나며 섬세하고 내면 중심적인 성향을 보입니다. 낯선 환경보다는 익숙한 관계를 중시하며, 깊이 있는 사고와 분석력이 강점입니다.")
    else:
        results.append("눈 간격이 평균에 가까워 사람들과 조화를 잘 이루며, 상황에 따라 외향성과 내향성을 균형 있게 조절할 수 있는 성격입니다.")

    if jaw_ratio > 0.55:
        results.append("턱이 길게 발달하여 인내심과 책임감이 강하고 목표를 향해 끊임없이 나아가는 추진력을 지닌 사람입니다. 현실적인 판단력과 실행력이 뛰어나 꾸준함이 돋보입니다.")
    elif jaw_ratio < 0.45:
        results.append("턱이 짧은 편으로 감수성이 풍부하고 감정의 흐름에 민감한 성향입니다. 예술적이거나 창의적인 분야에 강점을 보이며, 타인에게 세심한 배려를 잘합니다.")
    else:
        results.append("턱의 비율이 평균적으로 균형 잡혀 있어 감성과 이성을 고르게 갖춘 성격입니다. 상황에 맞는 유연한 판단력과 온화한 대인 관계가 특징입니다.")

    if forehead_w > 0.5:
        results.append("이마가 넓어 사고력이 뛰어나고 논리적이며, 지적 호기심이 많은 타입입니다. 계획 세우기를 좋아하며, 학문적·분석적 분야에서 두각을 나타낼 수 있습니다.")
    elif forehead_w < 0.35:
        results.append("이마가 좁은 편으로 직관과 창의력이 뛰어나며, 순간적인 판단이나 아이디어로 승부하는 경향이 있습니다. 전통적인 방식보다는 자유롭고 독창적인 사고를 선호합니다.")

    if nose_len > 0.18:
        results.append("코가 크고 길게 뻗어 있어 강한 자존감과 주도적인 성향을 보입니다. 리더십이 뛰어나며, 현실적이고 재물과 명예에 대한 욕구가 강해 성취지향적인 인물입니다.")
    else:
        results.append("코가 짧거나 낮은 편으로 내면의 안정과 정서를 중시합니다. 경쟁보다는 조화와 안전을 선호하며, 타인과의 관계에서 따뜻함과 배려를 중시합니다.")

    if lip_thickness > 0.03:
        results.append("입술이 두꺼운 편으로 감정 표현이 풍부하고 친근하며 따뜻한 인상을 줍니다. 인간관계에서 소통과 배려에 능하고, 사교성과 유머감각이 뛰어납니다.")
    else:
        results.append("입술이 얇은 편으로 신중하고 절제력 있으며, 말보다는 행동으로 신뢰를 주는 스타일입니다. 낯가림이 있지만, 깊이 있는 관계를 중요시합니다.")

    if asym < 0.01:
        results.append("얼굴 좌우 대칭이 뛰어나 안정감 있고 신뢰를 주는 인상을 줍니다. 외모에 대한 자신감이 높고, 사회적인 활동이나 대중과의 관계에서도 두각을 나타냅니다.")
    else:
        results.append("얼굴 좌우가 다소 비대칭적이지만, 이는 오히려 독특한 매력과 개성을 드러냅니다. 예술적 감성과 창의력이 뛰어나며, 틀에 얽매이지 않는 자유로운 사고방식을 지녔습니다.")

    score = calculate_beauty_score(asym, eye_ratio, jaw_ratio, forehead_w, nose_len, lip_thickness)

    return {
        "analysis": results,
        "score": score 
    }
