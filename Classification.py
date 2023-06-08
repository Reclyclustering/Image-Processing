import cv2
import RPi.GPIO as GPIO
import tensorflow as tf

# GPIO 핀 번호 설정
metal_led_pin = 19
paper_led_pin = 23
plastic_led_pin = 24

# GPIO 초기화
GPIO.setmode(GPIO.BCM)
GPIO.setup(metal_led_pin, GPIO.OUT)
GPIO.setup(paper_led_pin, GPIO.OUT)
GPIO.setup(plastic_led_pin, GPIO.OUT)

# 학습된 모델 로드
model = tf.saved_model.load('./SavedModel').signatures['serving_default']  # 모델 로드 방식 및 서명 이름 수정

# 분류할 클래스 레이블 정의
class_labels = ['metal', 'paper', 'plastic']  # 분류할 클래스 레이블 리스트

# 카메라 모듈 초기화
camera = cv2.VideoCapture(0)  # 0은 기본 카메라 장치를 의미합니다.

# 이미지 분류 함수
def classify_image(image):
    # 전처리 작업 (필요에 따라 변경 가능)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (384, 512))  # 모델에 맞는 입력 크기로 조정
    image = image.astype('float32')  # 이미지의 dtype을 float32로 변경
    image = image / 255.0  # 이미지를 0-1 범위로 정규화

    # 예측
    image = tf.expand_dims(image, axis=0)  # 배치 차원 추가
    predictions = model(tf.constant(image))  # 모델 호출 방식 변경
    class_index = tf.argmax(predictions['dense_1'], axis=1)[0].numpy()  # 예측에 사용된 서명 및 레이어 이름 수정
    class_label = class_labels[class_index]

    return class_label

try:
    # 카메라에서 이미지 가져오기 및 분류
    while True:
        ret, frame = camera.read()  # 카메라에서 프레임 읽기

        if not ret:
            print("카메라에서 이미지를 가져오는 데 문제가 발생했습니다.")
            break

        class_label = classify_image(frame)
        print("이미지 분류 결과:", class_label)

        # LED 제어
        GPIO.output(metal_led_pin, GPIO.LOW)  # 모든 LED를 끔
        GPIO.output(paper_led_pin, GPIO.LOW)
        GPIO.output(plastic_led_pin, GPIO.LOW)

        if class_label == 'metal':
            GPIO.output(metal_led_pin, GPIO.HIGH)  # metal 분류값에 해당하는 LED를 켬
        elif class_label == 'paper':
            GPIO.output(paper_led_pin, GPIO.HIGH)  # paper 분류값에 해당하는 LED를 켬
        elif class_label == 'plastic':
            GPIO.output(plastic_led_pin, GPIO.HIGH)  # plastic 분류값에 해당하는 LED를 켬

except KeyboardInterrupt:
    print("프로그램을 종료합니다.")

finally:
    # 정리 작업
    camera.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()  # GPIO 리소스 정리
