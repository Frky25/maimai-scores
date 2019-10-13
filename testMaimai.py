import os
from procMaimai import processImg

#['num_detect','ref_points', 'num_edges','title_processed']
def testAll(folder,imshows = []):
    scorepics = os.listdir(folder)
    for s in scorepics:
        print(s)
        results = processImg(folder+'/'+s,imshows)
        print(results)

def testFile(file,imshows = []):
    results = processImg(file,imshows)
    print(results)
