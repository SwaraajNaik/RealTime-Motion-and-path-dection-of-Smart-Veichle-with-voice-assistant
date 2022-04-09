import cv2
import numpy as np
import matplotlib.pylab as plt
import speech_recognition as sr  # pip install speechRecognition
import datetime  # pip install wikipedia# import webbrowser
import os
import smtplib
#import YOLO


def AcessCamera():          ##Access Camera
    cap = cv2.VideoCapture(0)
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(3,1000)
    cap.set(4,720)
    print(cap.get(3))
    print(cap.get(4))
    while(cap.isOpened()):
        ret,frame=cap.read()
        if ret ==True:
            gray=frame
            cv2.imshow('frame',gray)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
    cap.release()

    cv2.waitKey(500)
    cv2.destroyAllWindows()


def AdaptiveThresolding():          #Thresolding
    import cv2 as cv
    import numpy as np

    img = cv.imread('fogroad.jpg', 0)
    _, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2);
    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2);

    cv.imshow("Image", img)
    cv.imshow("THRESH_BINARY", th1)
    cv.imshow("ADAPTIVE_THRESH_MEAN_C", th2)
    cv.imshow("ADAPTIVE_THRESH_GAUSSIAN_C", th3)

    cv.waitKey(0)
    cv.destroyAllWindows()
finalimage=cv2.imread('blur-boy.PNG')


#-------------------------------------------------------------------------------------------------

def AdaptiveThresold1():  #ThresoldingVideo
    import cv2 as cv
    import numpy as np
    cap=cv.VideoCapture('fog1.mp4')
    cap.set(3,1000)
    cap.set(4,720)
    while cap.isOpened():
        ret,frame = cap.read()
        if ret ==True:
            frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            _, th1 = cv.threshold(frame,120, 255, cv.THRESH_BINARY)
            th2 = cv.adaptiveThreshold(frame, 230, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2);
            cv.imshow("frame",th2)
            if cv.waitKey(1) & 0xff == ord('q') :
                break
    cap.release()
    cv.waitKey(200)
    cv.destroyAllWindows()

def MotionDetection(): #MotionDetection
    import numpy as np
    import cv2

    cap = cv2.VideoCapture("fog1.mp4")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

    out = cv2.VideoWriter("output.mp4", fourcc, 5.0, (1100, 720))

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    print(frame1.shape)
    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 900:
                continue
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                          1, (0, 0, 255), 3)
        #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

        image = cv2.resize(frame1, (1280, 720))
        out.write(image)
        cv2.imshow("feed", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(40) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    out.release()

def LaneEdgeDetection():    #LaneLineDetection
    import matplotlib.pylab as plt
    import cv2
    import numpy as np

    def region_of_interest(img, vertices):
        mask = np.zeros_like(img)
        # channel_count = img.shape[2]
        match_mask_color = 255
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def drow_the_lines(img, lines):
        img = np.copy(img)
        blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2), (20, 255, 5), thickness=10)

        img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
        return img

    # = cv2.imread('road.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    def process(image):

        height = image.shape[0]
        width = image.shape[1]
        region_of_interest_vertices = [
            (1, height),
            (width /2, height /1.6),
            (width, height)
        ]
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        canny_image = cv2.Canny(gray_image, 100, 120)
        cropped_image = region_of_interest(canny_image,
                                           np.array([region_of_interest_vertices], np.int32), )
        lines = cv2.HoughLinesP(cropped_image,
                                rho=2,
                                theta=np.pi / 180,
                                threshold=40,
                                lines=np.array([]),
                                minLineLength=40,
                                maxLineGap=100)
        image_with_lines = drow_the_lines(image, lines)
        return image_with_lines

    cap = cv2.VideoCapture("clearroad.jpg")

    while cap.isOpened():
        ret, frame = cap.read()
        frame = process(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def LaneLineDetection():    #LaneLineDetection
    import matplotlib.pylab as plt
    import cv2
    import numpy as np

    def region_of_interest(img, vertices):
        mask = np.zeros_like(img)
        # channel_count = img.shape[2]
        match_mask_color = 255
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def drow_the_lines(img, lines):
        img = np.copy(img)
        blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2), (20, 255, 5), thickness=10)

        img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
        return img

    # = cv2.imread('road.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    def process(image):

        height = image.shape[0]
        width = image.shape[1]
        region_of_interest_vertices = [
            (1, height),
            (width /2, height /1.6),
            (width, height)
        ]
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        canny_image = cv2.Canny(gray_image, 100, 120)
        cropped_image = region_of_interest(canny_image,
                                           np.array([region_of_interest_vertices], np.int32), )
        lines = cv2.HoughLinesP(cropped_image,
                                rho=2,
                                theta=np.pi / 180,
                                threshold=40,
                                lines=np.array([]),
                                minLineLength=40,
                                maxLineGap=100)
        image_with_lines = drow_the_lines(image, lines)
        return image_with_lines

    cap = cv2.VideoCapture("emptyroad.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        frame = process(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
def EdgeLines():
    img = cv2.imread('fogroad.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    cv2.imshow('edges', edges)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow('image', img)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

def SmoothinigImage():
    img = cv2.imread('blur_boy.PNG')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    kernel = np.ones((5, 5), np.float32)/25
    dst = cv2.filter2D(img, -1, kernel)
    blur = cv2.blur(img, (5, 5))
    gblur = cv2.GaussianBlur(img, (5, 5), 0)
    median = cv2.medianBlur(img, 5)
    bilateralFilter = cv2.bilateralFilter(img, 9, 75, 75)

    titles = ['image', '2D Convolution', 'blur', 'GaussianBlur', 'median','FinalImage']
    images = [img, dst, blur, gblur, median, bilateralFilter,finalimage]

    for i in range(7):
        plt.subplot(3, 2, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()

#SmoothinigImage()


#AdaptiveThresolding()
#LaneEdgeDetection()
#EdgeLines()
#LaneLineDetection()
#AdaptiveThresold1()
#MotionDetection()

