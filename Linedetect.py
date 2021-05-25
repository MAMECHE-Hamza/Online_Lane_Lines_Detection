#!/usr/bin/env python3
import os
from datetime import datetime
import sys
import cv2 as cv
import numpy as np
from numpy.polynomial.polynomial import Polynomial as poly
#import pygame
#from pygame.locals import DOUBLEBUF
import sdl2
import sdl2.ext

class Display(object):
  def __init__(self, W, H):
    self.W, self.H,  = W, H
    #with sdl
    sdl2.ext.init()
    self.window = sdl2.ext.Window("Lane Detection", size=(self.W, self.H))
    self.window.show()
    self.surface = sdl2.ext.pixels3d(self.window.get_surface())
    
    
    #with pygame
    #pygame.init()
    #self.screen = pygame.display.set_mode((self.W, self.H), DOUBLEBUF)
    #self.surface = pygame.Surface(self.screen.get_size()).convert()

  def paint(self, img):
    # sdl version
    events = sdl2.ext.get_events()
    for event in events:
      if event.type == sdl2.SDL_QUIT:
        exit(0)
    
    self.surface[:,:,0:3] = img.swapaxes(0,1)
    self.window.refresh()
    
    #pygame verion
    #for event in pygame.event.get():
    #  pass
    
     # RGB, not BGR (might have to switch in twitchslam)
    #pygame.surfarray.blit_array(self.surface, img.swapaxes(0,1)[:, :, [0,1,2]])
    #self.screen.blit(self.surface, (0,0))

    #pygame.display.flip()

class CannyHough(object):
    def __init__(self, W, H):
        self.W, self.H,  = W, H
        self.canny_thresh = 140
        self.rho_res = 10
        self.theta_res = 1*np.pi/180
        self.hough_inter_min = 2  #min Hough lines intersection to consider 
        self.min_line_len = 2 #low thresh of line length 2
        self.max_line_gap = 6 #max dist between 2 points that intersect in hough to get connected in one line 6


    def process_frame(self, img):
        #canny
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_edges = cv.Canny(img_gray, self.canny_thresh, 3*self.canny_thresh)
        #region mask
        height = img_edges.shape[0]
        width = img_edges.shape[1]
        left_buttom = [0, height]
        right_buttom = [width, height]
        right_apex = [2.15/4*width, 4.15/7*height]
        left_apex =  [1.85/4*width, 4.15/7*height]
            # Fit lines using Polynomial.fit 
        left_fit = poly.fit((left_buttom[0], left_apex[0]), (left_buttom[1], left_apex[1]), 1, [])
        #print(left_fit)
        right_fit = poly.fit((right_buttom[0], right_apex[0]), (right_buttom[1], right_apex[1]), 1, [])
        #print(right_fit)
        #top_fit = poly.fit((left_buttom[0], right_buttom[0]), (left_buttom[1], right_buttom[1]), 0, [])
        X, Y = np.meshgrid(np.arange(0, width), np.arange(0, height), sparse = False, indexing = 'xy') # xy indexing by default means you should make it ij to get X = len(x)*len(y) otherwise you get X = len(y)*len(x)
        region_mask =  (Y > right_fit(X)) & (Y > left_fit(X)) & (Y > right_apex[1])
        img_edges[~region_mask] = 0
        #img_edges_3 = np.dstack((img_edges, img_edges, img_edges)) 
        #img_edges_3[~region_mask] = [0, 0, 0]
    
        #hough transform
        hough_lines = cv.HoughLinesP(img_edges, self.rho_res, self.theta_res, self.hough_inter_min, np.array([]), self.min_line_len, self.max_line_gap)
        img_lines = img*0
        for line in hough_lines:
            for x1,y1,x2,y2 in line:
                cv.line(img_lines, (x1,y1), (x2,y2), [100,0,255], 5, cv.LINE_AA)
        
        #img_final = cv.addWeighted(img_lines, 0.2, img, 0.8, 0)
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

  # create 2-D display
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