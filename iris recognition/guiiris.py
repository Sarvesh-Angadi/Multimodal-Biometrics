# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:43:30 2019

@author: HP
"""

import cv2
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import PIL
import os
#from tkinter.filedialog import askopenfilename
from PIL import ImageTk,Image 
#import cv2
import numpy as np
#from guisegmented import Whenpressed
import matlab.engine
#from frommatsingle import funsingle
def select_image():
    global panelA, panelB,panelC
 
    # open a file chooser dialog and allow the user to select an input
    # image
    path = filedialog.askopenfilename()
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path)
 
        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
        # convert the images to PIL format...
        image = Image.fromarray(image)
        
 
        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
    if panelA is None:
            # the first panel will store our original image
            panelA = Label(image=image, text="Blue Text in Verdana bold",fg = "blue",bg = "white",font = "Verdana 16 bold")
            x =Label(text="Eye",fg = "white",bg = "black",font = "Verdana 14 bold")
            panelA.image = image
            panelA.pack(side="top", padx=10, pady=10)
            x.pack(side="top")
 
            
 
        # otherwise, update the image panels
    else:
            # update the pannels
            panelA.configure(image=image)
            
            panelA.image = image
            
    
    
    

    
    print(path)
    eng = matlab.engine.start_matlab()
    filepath=path
    print(type(path))
    [c,dddd,circleiris, circlepupil, imagewithnoise] = eng.segmentirissinglegui(filepath,nargout=5)
    aa=[]#irrs
    dd=[]#pupil
    print(dddd)
    def funsingle():
        for i in range(5,len(c[0]),3):
            aa.append([c[0][i-2],c[0][i-1],c[0][i]])
            dd.append([dddd[0][i-2],dddd[0][i-1],dddd[0][i]])
        return(aa,dd,filepath)
    aq=[]
#imagepath=Whenpressed

    aq,dq,imagepath=funsingle()
    print(aq)
    print(dq)
#img2=[]
#for j in os.listdir(imagepath):
    img1=cv2.imread(imagepath) 
    yc, xc, r =dq[0] #radius of pupil
    yi,xi,ri=aq[0] #radius of iris
    # size of the image
    img=img1
    H, W ,hzzh= img.shape
    
    white_px = np.asarray([255, 255, 255])
    black_px = np.asarray([0, 0, 0])
    
    x, y = np.meshgrid(np.arange(W), np.arange(H)) #fr all points within the value W,and H,it creates a mesh or a grid 
    #u get [[0 1 2 .... 320]]
    # squared distance from the center of the circle
    d2 = (((x - xc)**2 + (y - yc)**2)) #for each point we get distance frm centre of pupil ,tried sqrt dint work as it needs only one value ,even this is fyn 
    #as a substitute 
    d3=(((x - xi)**2 + (y - yi)**2))#again same ,but this is for iris ,the dist of each point in grid from center coords of iris 
    #d2 and d3 are array of distances fr ecah combo of points 
    
    print(d2)
    mask = d2 < r**2# mask is True inside of the circle ,this is for pupil ,all distances less than the pupil diameter ,shuld be masked
    #with whichever color u wnat 
    mask1=d3>ri**2#same ,this is fr boundary outside the iris ,so watver is greater to boundary of this iris ,will get filled with color u want ,this is the
    #masking area
    

    
    average_color=255 #color wit which all maasks need to be filld 
    img[mask] = average_color #only that mask regoin is replaced with that avg color
    img[mask1]=average_color
    
#bytemask = np.asarray(mask*255, dtype=np.uint8)
#    inpainted = cv2.inpaint(img, bytemask, inpaintRadius=10, flags=cv2.INPAINT_TELEA) this part of code is needed if u wanna do more smoothening to the 
#    color u put in place of that region frm where u removed something ,basically the mask 
    cv2.imshow("segmented image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv_img = cv2.imread(filename)
    #if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        #image = cv2.imread(path)
 
        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
        # convert the images to PIL format...
        #img = img.fromarray(img)
    img=PIL.Image.fromarray(img)
    
        
        # ...and then to ImageTk format'
  
    img = ImageTk.PhotoImage(img)

    if panelB is None:
            # the first panel will store our original image
            panelB = Label(image=img,bg="red")
            y = Label(text="Segmented iris",fg="white",bg="red",font = "Verdana 12 bold")
            panelB.image = img
            y.pack(side = "left")
            panelB.pack(side="left", padx=10, pady=10)
            
 
            
 
        # otherwise, update the image panels
    else:
            # update the pannels
            panelB.configure(image=img)
            
            panelB.image = img 
    
    os.system('C:/Users/HP/Documents/USITv2.4.2/bin/caht -i {} -o C:/Users/HP/Documents/normalized/new.jpg'.format(path) )
    norm=cv2.imread("C:/Users/HP/Documents/normalized/new.jpg")

        
 
        # convert the images to PIL format...
    norm = PIL.Image.fromarray(norm)
        
 
        # ...and then to ImageTk format
    norm= ImageTk.PhotoImage(norm)
    if panelC is None:
            # the first panel will store our original image
            panelC = Label(image=norm,bg ="green")
            z= Label(text="Normalized iris",fg="white",bg="orange",font = "Verdana 12 bold")
            panelC.image = norm
            panelC.pack(side="right", padx=10, pady=10)
            z.pack(side="right")
            
 
 
        # otherwise, update the image panels
    else:
            # update the pannels
            panelC.configure(image=norm)
            
            panelC.image = norm
            
    
    
root = Tk()
root.title("IRIS PRE-PROCESSING")
panelA = None
panelB = None
panelC=None
 
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Select a new image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
 
# kick off the GUI
root.mainloop()


      
 

 
'''def select_image():
    global panelA, panelB,panelC
 
    # open a file chooser dialog and allow the user to select an input
    # image
    path = tk.filedialog.askopenfilename()
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path)
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #edged = cv2.Canny(gray, 50, 100)
 
        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
        # convert the images to PIL format...
        image = Image.fromarray(image)
        #edged = Image.fromarray(edged)
 
        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        #edged = ImageTk.PhotoImage(edged)
                # if the panels are None, initialize them
    if panelA is None or panelB is None or panelC is None:
            # the first panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
 
            # while the second panel will store the edge map
            panelB = Label(image=edged)
            panelB.image = edged
            panelB.pack(side="right", padx=10, pady=10)
            
            panelC = Label(image=edged)
            panelC.image = edged
            panelC.pack(side="right", padx=10, pady=10)
 
        # otherwise, update the image panels
    else:
            # update the pannels
            panelA.configure(image=image)
            panelB.configure(image=edged)
            panelC.configure(image=edged)
            panelA.image = image
            panelB.image = edged
            panelC.image = edged
root = Tk()
panelA = None
panelB = None
panelC=None
 
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
 
# kick off the GUI
root.mainloop()'''