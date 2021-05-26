#!/usr/bin/env python3
import os
from datetime import datetime
import sys
import cv2 as cv
import numpy as np
from numpy.polynomial.polynomial import Polynomial as poly
import sdl2
import sdl2.ext

class Display(object):
  def __init__(self, W, H):
    self.W, self.H,  = W, H

    sdl2.ext.init()
    self.window = sdl2.ext.Window("Lane Detection", size=(self.W, self.H))
    self.window.show()
    self.surface = sdl2.ext.pixels3d(self.window.get_surface())

  def paint(self, img):
    events = sdl2.ext.get_events()
    for event in events:
      if event.type == sdl2.SDL_QUIT:
        exit(0)
    
    self.surface[:,:,0:3] = img.swapaxes(0,1)
    self.window.refresh()

class CannyHough(object):
    def __init__(self, W, H):
        self.W, self.H,  = W, H
        self.canny_thresh = 140
        self.rho_res = 1
        self.theta_res = 1*np.pi/180
        self.hough_inter_min = 10  #min Hough lines intersection to consider 
        self.min_line_len = 5 #low thresh of line length 2
        self.max_line_gap = 10 #max dist between 2 points that intersect in hough to get connected in one line 6


    def process_frame(self, img):

        # hsv color segmentation
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_yellow = np.array([5,100,100]) #5, 100, 100
        upper_yellow = np.array([70,255,255]) # 70, 255, 255

        # threshold the HSV image to get only blue colors
        yellow_mask = cv.inRange(img_hsv, lower_yellow, upper_yellow)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, img_gray = cv.threshold(img_gray, 210, 255, cv.THRESH_BINARY)
        img_gray = cv.bitwise_or(img_gray, yellow_mask)
        img_edges = cv.Canny(img_gray, 1*self.canny_thresh, 3*self.canny_thresh)

        # region mask
        height = img_edges.shape[0]
        width = img_edges.shape[1]
        left_buttom = [0, height]
        right_buttom = [width, height]
        right_apex = [2.15/4*width, 4.15/7*height]
        left_apex =  [1.85/4*width, 4.15/7*height]

        # fit lines using Polynomial.fit 
        left_fit = poly.fit((left_buttom[0], left_apex[0]), (left_buttom[1], left_apex[1]), 1, [])
        right_fit = poly.fit((right_buttom[0], right_apex[0]), (right_buttom[1], right_apex[1]), 1, [])
        X, Y = np.meshgrid(np.arange(0, width), np.arange(0, height), sparse = False, indexing = 'xy') # xy indexing by default means you should make it ij to get X = len(x)*len(y) otherwise you get X = len(y)*len(x)
        region_mask =  (Y > right_fit(X)) & (Y > left_fit(X)) & (Y > right_apex[1])
        img_edges[~region_mask] = 0
        img_gray[~region_mask] = 0
        yellow_mask[~region_mask] = 0

        # hough transform
        hough_lines = cv.HoughLinesP(img_edges, self.rho_res, self.theta_res, self.hough_inter_min, np.array([]), self.min_line_len, self.max_line_gap)
        img_lines = img*0
        for line in hough_lines:
            for x1,y1,x2,y2 in line:
                cv.line(img_lines, (x1,y1), (x2,y2), [100,0,255], 10, cv.LINE_AA)
        
        img_final = cv.bitwise_or(img_lines, img)
        return img_final

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("%s <video.mp4>" % sys.argv[0])
    exit(-1)

  disp = None
    
  capture = cv.VideoCapture(sys.argv[1])

  # camera parameters
  W = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
  H = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
  COUNT = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

  if W > 1024:
    downscale = 1024.0/W
    H = int(H * downscale)
    W = 1024
  print("using camera %dx%d " % (W,H))

  # display
  if os.getenv("HEADLESS") is None:
    disp = Display(W, H)
  cannyhough = CannyHough(W, H)
  fourcc = cv.VideoWriter_fourcc(*'MP4V')
  video = cv.VideoWriter(f'{os.path.splitext(sys.argv[1])[0]}_Lanes_{datetime.now().time().second}.mp4', fourcc, 24, (W,H))
  i = 0
  while capture.isOpened():
    ret, frame = capture.read()
    try:
      frame = cv.resize(frame, (W, H))
    except:
      video.release()
      break

    print("\n*** frame %d/%d ***" % (i, COUNT))
    if ret == True:
      img = cannyhough.process_frame(frame)
      video.write(img)
      if disp is not None:
           disp.paint(img)
    else:
      break
    i += 1
  
  video.release()