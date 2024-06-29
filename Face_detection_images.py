import cv2


face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cv2.namedWindow('image')
img=cv2.imread('faces.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=face_detector.detectMultiScale(gray,1.1,3)
for (x,y,w,h) in faces:
  cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
