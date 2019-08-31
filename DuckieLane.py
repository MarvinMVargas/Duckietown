#!/usr/bin/env python
# coding: utf-8

# In[193]:


#!/usr/bin/env python
#Code written by Marvin M. Vargas Flores, July 2019 for Accenture DTIS.
#Duckiebot 2 lines lane follower 
#Version 06/08/2019
#
#
#Include opencv and numpy modules
import cv2 
import numpy as np
#Module for advanced mathematical operations
import math


# In[194]:


#Define black frames thickness in pixels
up_thic = 180
right_thic = 25
left_thic = 25
down_thic = 20

#Define threshold values of color filters
low_yellow = np.array([0,50,150])
up_yellow = np.array([150,255,255])
low_white = np.array([0,0,235])
up_white = np.array([255,255,255])
#Scale of the image, where 1 is 640*480
scale = 0.25
#Variable for the biggest contour area and the index of the contour in the contour array
global ybc,wbc
ybc = np.array([0,0])
wbc = np.array([0,0])
#Declare the font to write on the image
font = cv2.FONT_HERSHEY_SIMPLEX
#Declare contour boxes
global ybox,wbox
ybox = 0
wbox = 0
#Declare coordinate variables 
#Yellow line center coordinates
global ycx,ycy
ycx = 0
ycy = 0
#White line center coordinates
global wcx,wcy
wcx = 0
wcy = 0
#Lane center coordinates
global cx,cy
cx = 0
cy = 0
#Line slopes
global wsl,ysl
wsl = 0
ysl = 0
#X coordinate data from past frames centers
global pwbc,pybc
pcx = [0,0,0,0,0,0,0,0]
wpcx = 0
ypcx = 0
pwbc = [0,0]
pybc = [0,0]
#J is the counter for analyzed frames
j=0
k=0
#I is the counter for elements in contour arrays
yi=0
wi=0
#Define the amount of degrees for the acceptance of the distance in x axis
#between past center and new center
theta = 30

cap = cv2.VideoCapture('C:/Users/mARVIN/Downloads/duckie.mp4')


# In[195]:


def image_cut(frame):
    #Add a black frame to the image to close contours
    frame[0:int(up_thic*scale),0:int(640*scale)] = [0,0,0]
    frame[int(480*scale-down_thic*scale):int((480*scale)+8),0:int(640*scale)] = [0,0,0]
    frame[0:int((480*scale)+8),0:int(left_thic*scale)] = [0,0,0]
    frame[0:int((480*scale)+8),int(640*scale-right_thic*scale):int(640*scale)] = [0,0,0]
    pts = np.array([[0,80],[0,0],[160,0],[160,80],[120,50],[45,50]])
    pts.reshape((-1,1,2))
    cv2.fillPoly(frame,[pts],0)
    return frame


# In[196]:


def get_color_binary(frame):
    #Turn image from BGR to HSV space color
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #Create color mask 
    yellow_mask = cv2.inRange(hsv,low_yellow,up_yellow)
    white_mask = cv2.inRange(hsv,low_white,up_white)

    #Use a bitwise XOR to make sure white does not see yellow and
    #yellow does not see white
    white_and = cv2.bitwise_and(white_mask,yellow_mask,mask= yellow_mask)
    white_xor = cv2.bitwise_xor(white_mask,white_and)
    yellow_and = cv2.bitwise_and(yellow_mask,white_mask,mask= white_mask)
    yellow_xor = cv2.bitwise_xor(yellow_mask,yellow_and)
    #Apply color filter to original image
    yellow_res = cv2.bitwise_and(hsv,hsv,mask= yellow_xor)
    white_res = cv2.bitwise_and(hsv,hsv,mask= white_xor)

    #Turn image to grayscale
    yellow_bgr = cv2.cvtColor(yellow_res,cv2.COLOR_HSV2BGR)
    white_bgr = cv2.cvtColor(white_res,cv2.COLOR_HSV2BGR)
    yellow_gray = cv2.cvtColor(yellow_bgr,cv2.COLOR_BGR2GRAY)
    white_gray = cv2.cvtColor(white_bgr,cv2.COLOR_BGR2GRAY)

    #Turn mask to binary
    mret,yellow_mt = cv2.threshold(yellow_gray,125,255,cv2.THRESH_BINARY)
    wmret,white_mt = cv2.threshold(white_gray,125,255,cv2.THRESH_BINARY)
    return yellow_mt,white_mt,yellow_res,white_res
        

        


# In[197]:


def get_color_cont(yellow_mt,white_mt):
    #Use morphological dilation to reduce noise in the line
    kernel = np.ones((5,5),np.uint8)
    light_kernel = np.ones((2,2),np.uint8)
    yellow_kernel = np.ones((1,1),np.uint8)
    yellow_erode = cv2.erode(yellow_mt,yellow_kernel,iterations = 1)
    white_erode = cv2.erode(white_mt,light_kernel,iterations = 1)
    yellow_dilate = cv2.dilate(yellow_erode,kernel,iterations= 1)
    white_dilate = cv2.dilate(white_erode,kernel,iterations = 1)
    #yellow_close = cv2.morphologyEx(yellow_mt,cv2.MORPH_CLOSE,kernel)
    #white_close = cv2.morphologyEx(white_mt,cv2.MORPH_CLOSE,kernel)

    #Obtain the canny edges
    yellow_edge = cv2.Canny(yellow_dilate,30,200,True,3)
    white_edge = cv2.Canny(white_dilate,30,200,True,3)

    #Do a final threshold of the canny edges to find contours
    fret,yellow_fthresh = cv2.threshold(yellow_edge,100,255,cv2.THRESH_BINARY)
    wfret,white_fthresh = cv2.threshold(white_edge,100,255,cv2.THRESH_BINARY)

    #Read the contours 
    yellow_contours,hierarchy = cv2.findContours(yellow_fthresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    white_contours,whierarchy = cv2.findContours(white_fthresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    return yellow_contours,white_contours,yellow_fthresh,white_fthresh,yellow_dilate,white_dilate,yellow_edge,white_edge


# In[203]:


def get_lanes(yellow_contours,white_contours):
    global ycx,ycy,wcx,wcy,pwbc,pybc,ybc,wbc
    #Reset biggest contour
    ybc = [0,0]
    #Reset counter variable
    yi = 0

    #Check if any contour is found
    if len(yellow_contours):

        #Create for loop to run over all the countours
        for c in yellow_contours:
            #Check if the actual contour area is the biggest
            yx,yy,yw,yh = cv2.boundingRect(c)
            yM = cv2.moments(c)
            try:
                ycx = int(yM['m10']/yM['m00'])
            except:
                continue
                
            if wcx != 0:
                if yh>ybc[0] and ycx<=wcx-200*scale:
                    #Actual contour area is new biggest
                    ybc = [yh,yellow_contours[yi]]
            else:
                if yh>ybc[0]:
                    #Actual contour area is new biggest
                    ybc = [yh,yellow_contours[yi]]

            #Add 1 to the i variable to keep counting the index
            yi+=1
    else:
        ybc = pybc


 #Reset counter variable
    wi = 0
    #Reset array of biggest contour
    wbc = [0,0]

    #Check if any white_contours is found
    if len(white_contours):

        #Create for loop to run over all the countours
        for c in white_contours:
            #Check if the actual contour area is the biggest
            wx,wy,ww,wh = cv2.boundingRect(c)
            wM = cv2.moments(c)
            try:
                wcx = int(wM['m10']/wM['m00'])
            except:
                continue
                
            if (wh>wbc[0] and wcx>ycx+200*scale):
                #Actual contour area is new biggest
                wbc = [wh,c]
            #Add 1 to the i variable to keep counting the index
            wi+=1
    else:
        wbc = pwbc
    print("wbc:{}".format(wbc[0]))
    return ybc,wbc


# In[199]:


def draw_indicators(ycx,ycy,wcx,wcy,wcnt,ycnt,cx,cy):
    global frame,ybc,wbc,wsl,ysl
    if ybc[0]:
        frame = cv2.drawContours(frame,[ycnt],-1,(0,0,255),3)

        
        
    if wbc[0]:
        frame = cv2.drawContours(frame,[wcnt],-1,(0,0,255),3)
        
    cv2.circle(frame,(wcx,wcy),10,(0,255,0),-1)
    cv2.circle(frame,(ycx,ycy),10,(0,255,0),-1)
    cv2.circle(frame,(cx,cy),10,(255,0,0),-1)
    cv2.circle(frame,(int(320*0.25),cy),10,(0,0,255),-1)


# In[200]:


def lane_center(ybc,wbc):
    global frame,ycx,ycy,wcx,wcy,ybox,wbox
    #Reset counter variable
    #Draw in red the biggest contour in the image
    rows,cols = frame.shape[:2]
    if ybc[0]:

        #Find the moments of the biggest contour
        yM = cv2.moments(ybc[1])
        
        #Find centroid of the contour
        try:
            ycx = int(yM['m10']/yM['m00'])
            ycy = int(yM['m01']/yM['m00'])
        except:
            ycx = ycx
            ycy = ycy

    if wbc[0]:
        wM = cv2.moments(wbc[1])

        #Find centroid of the contour
        try:
            wcx = int(wM['m10']/wM['m00'])
            wcy = int(wM['m01']/wM['m00'])  
        except Exception as e:
            print(e)
            pass

    else:
        wcx = int(ycx+400*scale)
        wcy = int(ycy+400*scale)

    return ycx,ycy,wcx,wcy,ybox,wbox
        


# In[173]:


while (cap.isOpened()):
    global frame
    ret,frame = cap.read()
        
############################################################################################################################
######################################## 1st Part: Image filtering and operations ##########################################
############################################################################################################################
    if(ret):
            frame = image_cut(frame)
            if(k%10 == 0):    
                yellow_mt,white_mt,yellow_res,white_res = get_color_binary(frame)
                yellow_contours,white_contours,yellow_fthresh,white_fthresh,yellow_close,white_close,yellow_edge,white_edge = get_color_cont(yellow_mt,white_mt)
                
            #Contours is the original image with the contours marked on green
            cv2.imshow('contours',frame)
            #Final_thresh is the binary image where we look for the contours
            cv2.imshow('Yellow_Final_thresh',yellow_fthresh)
            cv2.imshow('White_Final_thresh',white_fthresh)
            #Dilation is the binary image after processing the color filtered image
            cv2.imshow('Yellow_Morphological ',yellow_close)
            cv2.imshow('White_Morphological',white_close)
            #Mask is the color filtered image
            cv2.imshow('Yellow_Mask',yellow_res)
            cv2.imshow('White_Mask',white_res)
            cv2.waitKey(0)       
            k = k+1
    else:
            break
            
cap.release()
cv2.destroyAllWindows()


# In[202]:


while (cap.isOpened()):
    global frame
    ret,frame = cap.read()
        
############################################################################################################################
######################################## 1st Part: Image filtering and operations ##########################################
############################################################################################################################
    if(ret):
            frame = image_cut(frame)
            if(k%10 == 0):    
                yellow_mt,white_mt,yellow_res,white_res = get_color_binary(frame)
                yellow_contours,white_contours,yellow_fthresh,white_fthresh,yellow_close,white_close,yellow_edge,white_edge = get_color_cont(yellow_mt,white_mt)

############################################################################           #############################################
####################################### 2nd Part: Image analysis to find line centers ###################################
#########################################################################################################################
                ybc,wbc = get_lanes(yellow_contours,white_contours)
                ycx,ycy,wcx,wcy,ybox,wbox = lane_center(ybc,wbc)

        
############################################################################################################################
######################################## 3rd Part: Image Analysis to find lane center ######################################
############################################################################################################################

            
                #Print line centers to keep track of data
                print('ycx:{0}  ycy:{1} wcx:{2} wcy:{3} \n'.format(ycx,ycy,wcx,wcy)) 
                cx = (int)((wcx + ycx)/2)
                cy = (int)((wcy + ycy)/2)

                #Print lane center to keep track of data
                print('cx:{0} cy:{1}'.format(cx,cy))

                #Calculate the threshold for accceptable distance, in x axis, from
                #old center to new center
                xdist = (128-cy)/math.tan(math.radians(theta))
                #Check if detected center makes sense with past data
                #J is the amount of frames analyzed, 7 is the size of the past data array
                if(j<=7):
                    #if j is bigger than one we have more than one data to compare
                    #if the difference 
                        if(j>1 and abs(pcx[j-1]-cx)>xdist):
                                if(pcx[j-1]-pcx[j-2]>=0):
                                        cx = pcx[j-1]-abs(pcx[j-1]-pcx[j-2])
                                else:
                                        cx = pcx[j-1]+abs(pcx[j-1]-pcx[j-2])

                        elif(j==1 and abs(pcx[j-1]-cx)>xdist):
                                cx = pcx[j-1]

                        pcx[j] = cx

                else:
                        if(abs(pcx[7]-cx)>xdist):
                                if(pcx[7]-pcx[6]>=0):
                                        cx = pcx[7] - abs(pcx[7]-pcx[6])
                                else:
                                        cx = pcx[7] + abs(pcx[7]-pcx[6])

                        for k in range(0,8):
                            if(k<7):
                                pcx[k] = pcx[k+1]
                            else:
                                pcx[k] = cx

                #Draw lane center
                #cv2.circle(frame,(cx,cy),10,(255,0,0),-1)
                #Draws the real center of the robot
                #cv2.circle(frame,(int(320*scale),cy),10,(0,0,255),-1)

                #Calculate the twist message
                if(cx<=320*scale):
                    dx = (320*scale)-cx
                    ang = dx/(320*scale)
                    linx = 1-ang
                else:
                    dx = (320*scale)-cx
                    ang = dx/(320*scale)
                    linx = 1-abs(ang)

                print('linx:{0} ang:{1} \n'.format(linx,ang))

                #Write the twist message in the image
                #cv2.putText(frame,'({0},0,0),(0,0,{1})'.format(linx,ang),(0,30), font, 1, (200,255,155), 2, cv2.LINE_AA)

            draw_indicators(ycx,ycy,wcx,wcy,wbc[1],ybc[1],cx,cy)
            #Contours is the original image with the contours marked on green
            cv2.imshow('contours',frame)
            #Final_thresh is the binary image where we look for the contours
            cv2.imshow('Yellow_Final_thresh',yellow_fthresh)
            cv2.imshow('White_Final_thresh',white_fthresh)
            #Dilation is the binary image after processing the color filtered image
            cv2.imshow('Yellow_Morphological ',yellow_edge)
            cv2.imshow('White_Morphological',white_edge)
            #Mask is the color filtered image
            cv2.imshow('Yellow_Mask',yellow_res)
            cv2.imshow('White_Mask',white_res)
            cv2.waitKey(0)       
            pwbc=wbc
            pybc=ybc
            k = k+1
            if j<=7:
                j+1
    else:
            break
            
cap.release()
cv2.destroyAllWindows()


# In[ ]:




