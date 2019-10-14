#Maimai results screen parser
This is a script to extract important information from pictures of Maimai results screens, for use in whatever kind of tracker you want to create.
##Setup
To get all the Python dependencies for this project, simply run
```
pip install -r requirements.txt
```

This project requires Tesseract https://github.com/tesseract-ocr/tesseract to be installed on the machine. If the tesseract installation is not automatically found by the script, you can add a config.ini file with contents:
```
[DEFAULT]
tesseract_path = <full/path/to/executable/tesseract>
```
Which will show the script where to find it.
Your Tesseract installation will need both the eng.traineddata and jpn.traineddata files (eng should be installed by default).
TODO: explain how to get these on different systems

This project requires OpenCV to be installed on the machine. 
TODO: point to OpenCV installation details

##Proccess Images
The main function for processing is processImg. It takes in an image as an array (e.g. from cv2.imread) and returns a python dictionary with the extracted data. It also takes an optional argument (imshow) with a list of strings for displaying some partially processed images for debugging purposes. The valid imshow strings are:
-`'ref_points'`: Show the detected reference points and the screen bounding circle. These are used to determine the cropping for detecting other features
-`'num_edges'`: Show the edge only version of the score, used for matching
-`'num_detect'`: Show the detected numbers and their locations. Number value corresponds with resistor color codes
-`'title_processed'`: Show the fully processed versions of the title. This is what is fed into OCR

So an example call might be:
```python
#process the image and display the reference points and processed titles
result = processImg(image,imshow=['ref_points','title_processed'])
```
###Image format
Results screen images to be processed by this should contain the entire screen (the circle of the screen is used for a refrence point), and they should be fairly head-on (no perspective correction is done).
TODO: Add example images

##Tesing utilities
There are some utility functions in testMaimai.py that are useful for local testing and debugging.
`testFile` takes a file name in as a string + an optional imshow argument. It processes the file and prints the results.
`testAll` takes a directory name in as a string + an optional imshow argument. It processes all of the files in that folder, printing the file name and results for each of them.


