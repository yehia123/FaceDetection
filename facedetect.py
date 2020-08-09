import cv2

imagePath = "./img/jessica-wilson-nx3N6enkY_k-unsplash.jpg"
cascPath = "./face_cascade.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

cap = cv2.VideoCapture(0)

while True:
    _,image = cap.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    print("Found {} faces".format(len(faces)))

    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 4)

    cv2.imshow("Faces found", image)
    
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
