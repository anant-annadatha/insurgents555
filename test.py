import numpy as np
import cv2
import imutils
import sys
import pytesseract
import pandas as pd
import time
import pymongo

image = cv2.imread('car2.jpeg')

image = imutils.resize(image, width=500)

cv2.imshow("Original Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("1 - Grayscale Conversion", gray)

gray = cv2.bilateralFilter(gray, 11, 17, 17)
#cv2.imshow("2 - Bilateral Filter", gray)

edged = cv2.Canny(gray, 170, 200)
#cv2.imshow("4 - Canny Edges", edged)

cnts,hierachy=cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] 
NumberPlateCnt = None 

count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  
            NumberPlateCnt = approx 
            break

# Masking the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
new_image = cv2.bitwise_and(image,image,mask=mask)
cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
cv2.imshow("Final_image",new_image)

# Configuration for tesseract
config = ('-l eng --oem 1 --psm 3')

# Run tesseract OCR on image
text = pytesseract.image_to_string(new_image, config=config)
spaces_rm = text.replace(" ", "")
spaces_trim = spaces_rm.strip()
#Data is stored in CSV file
raw_data = {'date': [time.asctime( time.localtime(time.time()) )], 
        'v_number': [spaces_trim]}

df = pd.DataFrame(raw_data, columns = ['date', 'v_number'])
df.to_csv('data.csv')

# Print recognized text
print(spaces_trim)

#Database Instance
cluster = pymongo.MongoClient("mongodb+srv://DEMODOGS:DEMODOGS@demodogs-apad.4aejn.mongodb.net/?retryWrites=true&w=majority")
db = cluster["DFS_Customer"]
dlplate = db["dlplate"]

#Sending data to MongoDB

myquery = { "dlplate": spaces_trim }
mydoc = dlplate.find(myquery)

if dlplate.find_one({ "dlplate": spaces_trim }):
        print('Returing User')
else:
        dlPlate_data = { "dlplate": spaces_trim}
        dlplate.insert_one(dlPlate_data)

#Listen and terminate until user presses any key 
cv2.waitKey(0)
