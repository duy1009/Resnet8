import numpy as np
import cv2
import glob

list_path = glob.glob(r"E:\Project3\Resnet8-Data\dataset\val\right\**")
t = 0
f = 0
for i, path in enumerate(list_path):
    
    img = cv2.imread(path)
    cv2.imshow("Test", img)
    key = cv2.waitKey(0)

    if key == ord("1"):
        t+=1
    elif key == ord("2"):
        f+=1
    elif key == ord("q"):
        break
    print(f"{i}/{len(list_path)} image: True {t} - False {f}")
