



import numpy as np
import cv2

name = input(u"Klasör ismi giriniz: ")
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
i=0
while(1):
    print(str(i))
    i = i + 1
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    
    
    kesilmis_kare=frame[0:300,0:300]
    fgmask = fgbg.apply(kesilmis_kare)
    
    
    
    cv2.imshow('frame',frame)
    cv2.imshow('kesilmis_kare',fgmask)
    cv2.imwrite('veriseti/' + (name) + '/' + str(i) + '.png', fgmask)
    
    cv2.waitKey(1) & 0xFF == ord ('q')
    if i>=250:
        break
cap.release()
cv2.destroyAllWindows()



