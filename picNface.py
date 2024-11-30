import cv2
import numpy as np
import face_recognition
import pandas as pd
import os
import time

# 기준 이미지 불러오기 및 인코딩
use_camera = True  # False면 img, True면 camera

# 신분증 이미지
#faceimg = face_recognition.load_image_file('face_pics/ai_hub_data/age/people4 (7).jpg')
faceimg = face_recognition.load_image_file('/Users/kim-jungeun/safeT_programs/face_pics/JE.jpeg')
faceimg = cv2.cvtColor(faceimg, cv2.COLOR_BGR2RGB)

# 테스트 이미지
imgTest = face_recognition.load_image_file('face_pics/ai_hub_data/age/people4 (8).jpg')
#imgTest = face_recognition.load_image_file('face_pics/21jungeun.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# 기준 이미지 얼굴 위치와 인코딩
faceLoc = face_recognition.face_locations(faceimg)[0]
encodeRef = face_recognition.face_encodings(faceimg)[0]
cv2.rectangle(faceimg, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

if use_camera:
    # 카메라 캡처 시작
    cap = cv2.VideoCapture(0)

# CSV 파일 존재 여부 확인 및 초기화
csv_filename = '/Users/kim-jungeun/safeT_programs/face_rec_data/face_features.csv' #본인 경로로 재설정 필요
if not os.path.exists(csv_filename):
    # CSV 파일 생성 및 헤더 추가
    df = pd.DataFrame(columns=[f'feature_{i}' for i in range(len(encodeRef))])
    df.to_csv(csv_filename, index=False)

start_time = time.time()
best_face_encoding = None
best_min_distance = float('inf')

while True:
    if use_camera:
        ret, frame = cap.read()
        if not ret:
            break
        # 프레임을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        rgb_frame = imgTest

    # 프레임에서 얼굴 인식 및 인코딩
    faceLocTest = face_recognition.face_locations(rgb_frame)
    encodeTest = face_recognition.face_encodings(rgb_frame, faceLocTest)

    for (top, right, bottom, left), face_encoding in zip(faceLocTest, encodeTest):
        # 기준 얼굴과 현재 프레임 얼굴 비교
        results = face_recognition.compare_faces([encodeRef], face_encoding)
        faceDis = face_recognition.face_distance([encodeRef], face_encoding)

        # 가장 작은 유사도 값 찾기
        if faceDis[0] < best_min_distance:
            best_min_distance = faceDis[0]
            best_face_encoding = face_encoding
            best_match_text = f'{results[0]} {round(best_min_distance, 2)}'

        # 얼굴 주위에 사각형 그리기
        if use_camera:
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)
        else:
            cv2.rectangle(imgTest, (left, top), (right, bottom), (255, 0, 255), 2)

    # 결과와 유사성을 이미지 상단에 명시
    if best_match_text:
        if use_camera:
            cv2.putText(frame, best_match_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(imgTest, best_match_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # N초가 지나면 유사도 값 저장(현재는 10초로 설정)
    if time.time() - start_time >= 10:
        if best_face_encoding is not None:
            if best_min_distance <= 0.40:#유사도 N이하면 얼굴 데이터 csv로 저장(현재는 0.4로 설정)
                print(f"Face recognized with distance: {best_min_distance}. Features saved.")
                # CSV 파일에 특징값 저장
                df = pd.read_csv(csv_filename)
                new_row = pd.DataFrame([best_face_encoding], columns=df.columns)
                if not new_row.isna().all().all():  # 새로운 행이 비어 있거나 모든 값이 NA가 아닌지 확인
                    df = pd.concat([df, new_row], ignore_index=True)
                    df.to_csv(csv_filename, index=False)
            else:
                print("동일인이 아니라고 판별되어 얼굴 데이터를 저장하지 않았습니다.")
        break

    # 결과 보여주기
    if use_camera:
        cv2.imshow('Camera', frame)
    else:
        cv2.imshow('Test Image', imgTest)

    cv2.imshow('Reference Face', faceimg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 및 윈도우 종료
if use_camera:
    cap.release()
cv2.destroyAllWindows()
