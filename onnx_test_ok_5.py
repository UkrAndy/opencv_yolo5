import onnxruntime as ort
import numpy as np
import cv2

# Завантаження моделі
session = ort.InferenceSession('models/yolov5s6_pose_640.onnx')
# session = ort.InferenceSession('models/yolov7-w6-pose-nms.onnx')


# Функція для передобробки зображення
# def preprocess(image):
#     # Зміна розміру зображення до розміру, який очікує модель (640x640)
#     image = cv2.resize(image, (640, 640))
#     # Перетворення зображення в формат, який очікує модель
#     image = image.astype(np.float32)
#     image = np.transpose(image, (2, 0, 1))  # Зміна порядку осей на (C, H, W)
#     image = np.expand_dims(image, axis=0)  # Додавання осі для батчу
#     return image

def preprocess(img, img_mean=127.5, img_scale=1 / 127.5):
    img = cv2.resize(img[:, :, ::-1], (640, 640), interpolation=cv2.INTER_LINEAR)
    img = (img - img_mean) * img_scale
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, 0)
    img = img.transpose(0, 3, 1, 2)
    return img

def  getaversge_mean(keys):
    # return np.mean([confidence for x, y, confidence in keys])
    return np.mean(keys[8::3])

# def preprocess(image):
#     # Зміна розміру зображення до розміру, який очікує модель (640x640)
#     image = cv2.resize(image, (640, 640))
#     # Перетворення зображення в формат, який очікує модель
#     image = image.astype(np.float32)
#     image = np.transpose(image, (2, 0, 1))  # Зміна порядку осей на (C, H, W)
#     image = np.expand_dims(image, axis=0)  # Додавання осі для батчу
#     return image

# Функція для відображення ключових точок
def draw_keypoints(image, keypoints, original_size):
    h, w = original_size
    print('=================')
    print(keypoints)
    print('-------------------------------')
    cv2.rectangle(image,
                  (int(keypoints[0]* w / 640),int(keypoints[1]* h / 640)),
                  (int(keypoints[2]* w / 640), int(keypoints[3]* h / 640)),
                  (0, 0, 255),
                  2
                  )

    for i in range(6, len(keypoints), 3):
        x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
        print(x, y, conf)
        if conf > 0.5 and 0<=x and 0<=y:  # Відображати тільки впевнені ключові точки
            x = int(x * w / 640)  # Масштабування координат назад до оригінальних розмірів
            y = int(y * h / 640)
            if i/3< 3:
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
            else:
                cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), -1)

# Отримання зображення з камери
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    original_size = frame.shape[:2]  # Зберігаємо оригінальні розміри

    # Передобробка зображення
    input_image = preprocess(frame)

    # Виконання передбачення
    inputs = {session.get_inputs()[0].name: input_image}
    # outputs = session.run(None, inputs)
    outputs = session.run([], inputs)

    # Обробка результатів
    keypoints = outputs[0][0]  # Припускаємо, що ключові точки в першому виході
    print('====>>> keypoints')



    for keys in outputs[0]:
        # draw_keypoints(frame, keys)
        print('===>>>*****   getaversge_mean(keys)')
        # print(keys)
        # conf_p =getaversge_mean(keys)
        conf_p =keys[4]

        print(f'{conf_p} - {getaversge_mean(keys)}')
        # keypoints = keys[:-1]
        confidence = keys[-1]
        print('----- confidence')
        print(len(keys))
        print(confidence)
        #
        if conf_p > 0.5 :
            draw_keypoints(frame, keys, original_size)
        # if len(keys)>10:
        #     draw_keypoints(frame, keys, original_size)

        # draw_keypoints(frame, keypoints, original_size)


    # draw_keypoints(frame, keypoints)
    # draw_keypoints(frame, outputs[0][1])

    # print(len(outputs[0]))

    # Відображення кадру
    cv2.imshow('ONNX Pose Detection', frame)

    # Вихід при натисканні клавіші 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Звільнення ресурсів
cap.release()
cv2.destroyAllWindows()
