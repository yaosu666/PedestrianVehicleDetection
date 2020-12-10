
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time
yolo = YOLO()

capture=cv2.VideoCapture(0)

fps = 0.0
while(True):
    t1 = time.time()
    
    ref,frame=capture.read()
    
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    frame = Image.fromarray(np.uint8(frame))

    frame = np.array(yolo.detect_image(frame))

    
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video",frame)


    c= cv2.waitKey(30) & 0xff 
    if c==27:
        capture.release()
        break
