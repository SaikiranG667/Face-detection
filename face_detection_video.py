import cv2

cap=cv2.VideoCapture(0)
face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while(cap.isOpened()):
  ret,frame=cap.read()
  gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

  faces=face_classifier.detectMultiScale(gray,1.1,3)
  for (x,y,w,h) in faces:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
  cv2.imshow("video",frame)
  if cv2.waitKey(1) & 0xFF==27:
    break
cap.release()
cv2.destroyAllWindows()