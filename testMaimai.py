import os
import cv2
from procMaimai import processImg

#['num_detect','ref_points', 'num_edges','title_processed']
def testAll(folder,imshows = []):
    scorepics = os.listdir(folder)
    for s in scorepics:
        print(s)
        #load image
        img = cv2.imread(folder+'/'+s,cv2.IMREAD_COLOR)
        results = processImg(img,imshows)
        print(results)

def testFile(file,imshows = []):
    #load image
    img = cv2.imread(file,cv2.IMREAD_COLOR)
    results = processImg(img,imshows)
    print(results)
