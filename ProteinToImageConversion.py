import os
import cv2
import text_to_image
import pandas as pd
def main():
    os.makedirs("multi-label-png")
    os.makedirs("multi-label")
    df=pd.read_csv("test-cc")
    for index,row in df.iterrows():
        text_to_image.encode(row["sequences"],"multi-label-png/"+row["proteins"])
        image=cv2.imread("multi-label-png/"+row["proteins"]+".png",0)
        resizedImage=cv2.resize(image,(299,299))
        cv2.imwrite("multi-label/"+row["proteins"]+".jpg",resizedImage)

if __name__ == '__main__':
    main()