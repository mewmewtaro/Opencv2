import cv2 as cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
path = "img_people/pic"

def create_dataset(img,id,img_id):
    cv2.imwrite(path + str(id)+'.'+str(img_id)+'.jpg',img)

def draw_boundary(img, Classifier, scaleFactor, minNeighbors, color, text):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    features = Classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
        coords = [x, y, w, h]
    return img, coords

def detect(img, faceCascade, img_id):
    img, coords = draw_boundary(img, faceCascade, 1.1, 10, (0, 255, 0), 'Face')
    id = 1
    if len(coords) == 4 :  #กรณีเจอใบหน้าและมีกรอบจับใบหน้า ขนาดของกรอบจะเป็น img(y:y+h,x:x+w)
        result = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
        create_dataset(result, id, img_id,)
    return img

img_id = 0
cap = cv2.VideoCapture('Somkiat.mp4')
while (True):
    ret, frame = cap.read()
    frame = detect(frame, faceCascade, img_id)
    cv2.imshow('frame', frame)
    img_id += 1
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()

