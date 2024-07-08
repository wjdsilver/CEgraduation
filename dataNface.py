import cv2
import numpy as np
import face_recognition
import pandas as pd
import os
import time

# use_camera 변수 설정
use_camera = False  # False면 img, True면 camera

# CSV 파일 경로
csv_filename = 'C:/KJE/CE_graduation/face_rec/face_rec_data/face_features.csv'

# 저장된 얼굴 데이터 불러오기
df = pd.read_csv(csv_filename)
saved_encodings = df.values

if use_camera:
    # 카메라 캡처 시작
    cap = cv2.VideoCapture(0)
else:
    # 테스트 이미지 불러오기
    #test_image_path = 'face_pics/21jungeun.jpg'  # 여기에 테스트 이미지 경로를 설정
    test_image_path = 'face_pics/ai_hub_data/age/people1 (3).jpg'  # 여기에 테스트 이미지 경로를 설정
    imgTest = face_recognition.load_image_file(test_image_path)
    imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

start_time = time.time()
min_distance = float('inf')
best_match_text = ""

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

    if not encodeTest:
        print("No face found in the frame.")
    else:
        for (top, right, bottom, left), face_encoding in zip(faceLocTest, encodeTest):
            # 저장된 얼굴 데이터와 현재 프레임 얼굴 비교
            faceDis = face_recognition.face_distance(saved_encodings, face_encoding)
            current_min_distance = np.min(faceDis)

            # 디버깅: 유사도 값 출력
            print(f"Face distance: {current_min_distance}")

            # 최솟값 갱신
            if current_min_distance < min_distance:
                min_distance = current_min_distance
                if min_distance <= 0.4:#유사도 N이하면 얼굴 데이터 csv로 저장(현재는 0.4로 설정)
                    best_match_text = "동일인입니다."
                else:
                    best_match_text = "동일인이 아닙니다."

            # 얼굴 주위에 사각형 그리기
            if use_camera:
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)
                #cv2.putText(frame, best_match_text, (left, top - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.rectangle(imgTest, (left, top), (right, bottom), (255, 0, 255), 2)
                #cv2.putText(imgTest, best_match_text, (left, top - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

    # N초가 지나면 루프 종료(현재는 5초로 설정)
    if time.time() - start_time >= 5:
        break

    # 결과 보여주기
    if use_camera:
        cv2.imshow('Camera', frame)
    else:
        cv2.imshow('Test Image', imgTest)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 최종 결과 출력
print(f"최종 결과: {best_match_text} (유사도 거리: {min_distance})")

# 카메라 및 윈도우 종료
if use_camera:
    cap.release()
cv2.destroyAllWindows()
