import cv2
import pandas as pd


face_model = cv2.face.LBPHFaceRecognizer_create()
face_model.read('facemodel.yml')

image_file = 'sunnee.jpg'

feature_size = (96, 96)

label_file = './labels.csv'
df = pd.read_csv(label_file)
y_label = df.name


def detect_faces(image):
    #     casc_file = "lbpcascade_frontalface.xml"
    casc_file = "haarcascade_frontalface_default.xml"
    frontal_face = cv2.CascadeClassifier(casc_file)
    bBoxes = frontal_face.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    return bBoxes


color_image = cv2.imread(image_file)
dimensions = color_image.shape
print(dimensions)
img_resize_factor = 800 / dimensions[1]

color_image = cv2.resize(color_image, None, fx=img_resize_factor, fy=img_resize_factor, interpolation=cv2.INTER_AREA)

gray_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

bBoxes = detect_faces(gray_frame)

for bBox in bBoxes:
    (p, q, r, s) = bBox
    cv2.rectangle(color_image, (p, q), (p + r, q + s), (25, 255, 25), 2)

    crop_image = gray_frame[q:q + s, p:p + r]

    crop_image = cv2.resize(crop_image, feature_size)  # ksb

    [pred_label, pred_conf] = face_model.predict(crop_image)
    print("Predicted person: {:8}".format(y_label[pred_label]))

    box_bg = (0, 255, 0)
    box_bg = (0, 180, 0)
    cv2.rectangle(color_image, (p, q), (p + 95, q - 22), box_bg, cv2.FILLED)

    box_text = y_label[pred_label][:7]
    txt_color = (255, 255, 255)
    cv2.putText(color_image, box_text, (p + 4, q - 4),
                cv2.FONT_HERSHEY_PLAIN, 1.3, txt_color, 2)

cv2.imwrite('pred_imgb.jpg', color_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
cv2.imshow("Win", color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()