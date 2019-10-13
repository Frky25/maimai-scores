import imutils
import cv2
import numpy as np
import os
import pytesseract
import json
import configparser
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def song_names():
    namesen = []
    namesjp = []
    with open('maimai.json','r', encoding='utf8') as read_file:
        data = json.load(read_file)
    for song in data:
        namesen.append(str(song['name']))
        namesjp.append(str(song['name_jp']))
    return namesen,namesjp

def largest_component(image):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    return img2

def find_digits(image,imshows=[]):
    #refrence digit height and width
    dW = 48
    dH = 67
    #range of scales to check for best match
    s1 = float(dH)/image.shape[0]
    s2 = s1*2.0

    #attempt to match all digits at all scale
    maxth = 0
    maxs = 0
    for scale in np.linspace(s1, s2, 100)[::-1]:
        edges = cv2.Canny(cv2.resize(image,None,fx=scale,fy=scale),300,500)
        for d in range(0,10):
            digit = cv2.imread('digits/'+str(d)+'.png',0)
            res = cv2.matchTemplate(edges,digit,cv2.TM_CCOEFF_NORMED)
            if np.amax(res)>maxth:
                maxth = np.amax(res)
                maxs = scale
    #take the scale of the best match found
    image = cv2.resize(image,None,fx=maxs,fy=maxs)
    edges = cv2.Canny(image,250,400)
    if "num_edges" in imshows:
        cv2.imshow('edges',edges)
    #result saved in the form (x,y),res,num for up to 5 digits
    results = [[(0,0),(0,0),(0,0),(0,0),(0,0)],[0.,0.,0.,0.,0.],[-1,-1,-1,-1,-1]]

    #check each digit for matches
    for d in range(0,10):
        digit = cv2.imread('digits/'+str(d)+'.png',0)
        res = cv2.matchTemplate(edges,digit,cv2.TM_CCOEFF_NORMED)
        threshold = 0.20
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            placed = False
            #check for overlapping and pick the better match
            for i in range(0,5): 
                rpt = results[0][i]
                if np.linalg.norm([pt[0]-rpt[0],pt[1]-rpt[1]])<30:
                    placed = True
                    if res[pt[1],pt[0]]>results[1][i]:
                        results[0][i] = pt
                        results[1][i] = res[pt[1],pt[0]]
                        results[2][i] = d
            if not placed:
                #not overlapping with any existing digits
                #replace the worst digit if better
                minRes = np.argmin(results[1])
                if res[pt[1],pt[0]]>results[1][minRes]:
                    results[0][minRes] = pt
                    results[1][minRes] = res[pt[1],pt[0]]
                    results[2][minRes] = d

    if "num_detect" in imshows:
        #based on resistor color codes
        colors = [(0,0,0),
                  (42,42,165),
                  (0,0,255),
                  (0,165,255),
                  (0,255,255),
                  (0,255,0),
                  (255,0,0),
                  (180,0,180),
                  (128,128,128),
                  (255,255,255)]
        for i in range(0,5):
            if results[2][i] >= 0:
                cv2.rectangle(image,
                          results[0][i],(results[0][i][0]+48,results[0][i][1]+67),
                          colors[results[2][i]],2)
        cv2.imshow('digits',image)

    #order the digits
    xcoords = np.array(results[0])[:,0]
    order = np.argsort(xcoords)
    digits = np.array(results[2])[order]
    if digits[1]>0:
        #can only be a 4 digit score
        digits = digits[1:]
    if -1 in digits:
        print("Score improperly detected")
        return 0

    result = ''
    for d in digits:
        result = result+str(d)
    result = int(result)
        
    return result

def read_title(image,imshows=[]):
    #resize for better OCR
    image = cv2.resize(image,None,fx=120./image.shape[0],fy=120./image.shape[0])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    #dynamically determine threshold levels
    hist,bins = np.histogram(gray.flatten(),256,[0,256])
    cdf = hist.cumsum()
    clip1 = 0
    clip2 = 0
    gray1 = gray.copy()
    gray2 = gray.copy()
    #two thresholds to help with differing amounts of text
    while(float(cdf[clip1])/gray.size<.02):
        clip1 += 1
    while(float(cdf[clip2])/gray.size<.03):
        clip2 += 1
    _,gray1 = cv2.threshold(gray1,clip1,255,cv2.THRESH_BINARY)
    _,gray2 = cv2.threshold(gray2,clip2,255,cv2.THRESH_BINARY)
    if 'title_processed' in imshows:
        cv2.imshow('title 0.02',gray1)
        cv2.imshow('title 0.03',gray2)

    #load tesseract location from config file, if it exists
    config = configparser.ConfigParser()
    try:
        config.read('config.ini')
    except FileNotFoundError:
        pass
    else:
        if 'tesseract_path' in config['DEFAULT']:
            pytesseract.pytesseract.tesseract_cmd = config['DEFAULT']['tesseract_path']

    #Search for english and japanese text in both thresholded images
    configen = ('-l eng --oem 1 --psm 7')
    configjp = ('-l jpn --oem 1 --psm 7')
    texten1 = pytesseract.image_to_string(gray1, config=configen)
    texten2 = pytesseract.image_to_string(gray2, config=configen)
    textjp1 = pytesseract.image_to_string(gray1, config=configjp)
    textjp2 = pytesseract.image_to_string(gray2, config=configjp)

    if 'title_processed' in imshows:
        #Detected text for debugging
        print("texten 0.02: " + texten1)
        print("texten 0.03: " + texten2)
        print("textjp 0.02: " + textjp1)
        print("textjp 0.03: " + textjp2)

    #Fuzzy match OCR results to song database
    #ratio match works for most songs
    #token_set_ratio helps for songs that are scrolled around
    songsen,songsjp = song_names()
    resen11,resen21,resen12,resen22,resjp11,resjp21,resjp12,resjp22 = [[],[],[],[],[],[],[],[]]
    if(len(texten1)>1):
        resen11 = process.extract(texten1,songsen,limit=4,scorer=fuzz.token_set_ratio)
        resen12 = process.extract(texten1,songsen,limit=4,scorer=fuzz.ratio)
    if(len(texten2)>1):
        resen21 = process.extract(texten2,songsen,limit=4,scorer=fuzz.token_set_ratio)
        resen22 = process.extract(texten2,songsen,limit=4,scorer=fuzz.ratio)
    if(len(textjp1)>1):
        resjp11 = process.extract(textjp1,songsjp,limit=4,scorer=fuzz.token_set_ratio)
        resjp12 = process.extract(textjp1,songsjp,limit=4,scorer=fuzz.ratio)
    if(len(textjp2)>1):
        resjp21 = process.extract(textjp2,songsjp,limit=4,scorer=fuzz.token_set_ratio)
        resjp22 = process.extract(textjp2,songsjp,limit=4,scorer=fuzz.ratio)

    #take the top 4 results from all the matching TODO: parameterize
    resUns = [['','','',''],[-1,-1,-1,-1],[0,0,0,0]]
    for results in [resen11,resen21,resen12,resen22,resjp11,resjp21,resjp12,resjp22]:
        for r in results:
            #find the song's index for easier information lookups
            song = r[0]
            index = -1
            if song in songsen:
                index = songsen.index(song)
            elif song in songsjp:
                index = songsjp.index(song)
            else:
                print("Song lookup in list failed. This shouldn't happen")
            #duplicate check
            if index in resUns[1]:
                i = resUns[1].index(index)
                if r[1] > resUns[2][i]:
                    #new score is better: update
                    resUns[2][i] = r[1]
            elif r[1]>min(resUns[2]):
                resUns[0][np.argmin(resUns[2])]=r[0]
                resUns[1][np.argmin(resUns[2])]=index
                resUns[2][np.argmin(resUns[2])]=int(r[1])
    #Sort the results and generate the output tuples
    order = np.argsort(resUns[2])[::-1]
    res = [('',-1,0),('',-1,0),('',-1,0),('',-1,0)]
    for i in range(0,4):
        res[i] = (resUns[0][order[i]],resUns[1][order[i]],resUns[2][order[i]])
    return res

def processImg(image,imshows=[]):
    results = {}
    
################### Score Detection ###################
    #resizing for consistent circle detection
    wid = 500.0
    factor = wid/image.shape[1]
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scaled_img = cv2.resize(image,None,fx=factor,fy=factor)
    grey_img = cv2.resize(grey_img,None,fx=factor,fy=factor)
    circles = cv2.HoughCircles(grey_img,cv2.HOUGH_GRADIENT,1,500,
                                param1=50,param2=100,minRadius=100,maxRadius=500)

    if circles is not None:
        #extract the most prominent circle (the screen)
        circles = np.uint16(np.around(circles))
        c = circles[0][0]
        cx = c[0]
        cy = c[1]
        cr = c[2]
        # draw the center of the circle
        cv2.circle(scaled_img,(cx,cy),2,(255,0,0),3)
        cv2.circle(scaled_img,(cx,cy),cr,(255,0,0),3)
    else:
        print("Screen not found")
        return

    #find the song title banner by searching for red on screen
    hsv = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,150,100])
    upper_red = np.array([5,255,220])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([165,150,100])
    upper_red = np.array([180,255,220])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    title_mask = cv2.bitwise_or(mask1,mask2)
    #clean up the mask
    for y in range(title_mask.shape[0]):
        for x in range(title_mask.shape[1]):
            if y > cy:
                title_mask[y,x]=0 #title shouldn't be below center
    k = np.ones((5,5),np.uint8)
    title_mask = cv2.dilate(title_mask,k,iterations = 2)
    title_mask = cv2.erode(title_mask,k,iterations=4)
    title_mask = largest_component(title_mask)

    #find the center of mass of the title mask
    M = cv2.moments(title_mask)
    MX = int(M["m10"] / M["m00"])
    MY = int(M["m01"] / M["m00"])    
    cv2.circle(scaled_img, (MX, MY), 4, (255, 255, 255), -1)
    if "ref_points" in imshows:
        cv2.imshow('detected circles',scaled_img)

    #using the title mask and circle center, crop out the score
    yDist = cy-MY
    scoreYMin = int((cy - yDist*1/2)*1/factor)
    scoreYMax = int((cy + yDist*1/2)*1/factor)
    scoreXMin = int((cx - yDist*1.25)*1/factor)
    scoreXMax = int((cx + yDist*1.5)*1/factor)
    score = image[scoreYMin:scoreYMax,scoreXMin:scoreXMax,:]
    #process the score
    results['score'] = find_digits(score,imshows)

################### Title Detection ###################

    #prepare and crop the image for title detection
    title_mask = cv2.dilate(title_mask,k,iterations = 2)
    title_mask = cv2.resize(title_mask,(image.shape[1],image.shape[0]))
    title_mask = np.uint8(title_mask)
    title_masked = cv2.bitwise_and(image,image,mask=title_mask)
    rect = cv2.boundingRect(title_mask)
    title_masked = title_masked[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
    
    if title_masked.shape[0]/title_masked.shape[1]>.25:
        #correction for songs on expert
        title_masked = title_masked[int(title_masked.shape[0]/2):title_masked.shape[0]]
    #crop in to remove bright stars from corners
    title_masked = title_masked[:,int(title_masked.shape[1]*.1):int(title_masked.shape[1]*.9)]

    results['titles'] = read_title(title_masked,imshows)

    return results
