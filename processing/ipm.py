import numpy as np
# -*- coding: utf-8 -*-
def calculateHMatrix (originalPoints, transformedPoints):
  #print ("calculating H matrix")
  
  orig = np.array([tuple(i) for i in originalPoints]);
  
  #print (orig)
  transformed = np.array([tuple(i) for i in transformedPoints]);
  #print (transformed)

  A = []
  for i in range (0,4):
    A.append([0, 0, 0, -orig[i][0], -orig[i][1], -1, transformed[i][1] * orig[i][0], transformed[i][1] * orig[i][1], transformed[i][1]])
    A.append([orig[i][0], orig[i][1], 1, 0, 0, 0, -transformed[i][0] * orig[i][0], -transformed[i][0] * orig[i][1], -transformed[i][0]])

  A = np.array(A)
  #print ("A: ")
  #print (A)

  _, _, vt = np.linalg.svd(A)
  #print ("u:"); print (u);
  #print ("d: "); print (d);
  #print ("vt: "); print (vt);
  h = vt[-1]
  #print ("H as vector: ", str(h))
  
  H = np.matrix([
    [h[0],h[1],h[2]],
    [h[3],h[4],h[5]],
    [h[6],h[7],h[8]]])

  return H

def transform_image(width, height, originalImage, transformationMatrix, enableInterpolation = True):

  Hinv = linalg.inv(transformationMatrix)
  transformedImage = array(Image.new (originalImage.mode, (width,height)))
  (originalWidth, originalHeight) = originalImage.size

  originalArray = array(originalImage)
  for y in range(0, height):
    #print ("Progress %.2lf %%" % (float(y)/float(height-1) * 100))
    for x in range (0, width):
      pointTransformed = np.matrix([[x], [y], [1]])
      pointOriginal = Hinv * pointTransformed
     
      t = [float(pointOriginal[0][0]/pointOriginal[2][0]),
        float(pointOriginal[1][0]/pointOriginal[2][0]),
        1]

      #print ("Point transformed: " +str(pointTransformed) + ", Point original: " + str (t) + '\n==============\n')
      
      xOrig = t[0]
      yOrig = t[1]
      

      if (enableInterpolation
        and (xOrig != int(xOrig) or yOrig != int(yOrig))
        and xOrig + 1 <= originalWidth
        and yOrig + 1 <= originalHeight):

        #print ("Interpolating (%f, %f)" % (xOrig, yOrig))

        xOrigInt = int (xOrig)
        yOrigInt = int (yOrig)

        dx = xOrig - xOrigInt
        dy = yOrig - yOrigInt

        #print ("dx: %f; dy: %f" % (dx, dy))

        point = (originalArray[yOrigInt][xOrigInt] * (1-dx)*(1-dy)
              + originalArray[yOrigInt][xOrigInt+1]*dx*(1-dy)
              + originalArray[yOrigInt+1][xOrigInt]*(1-dx)*dy
              + originalArray[yOrigInt+1][xOrigInt+1]*dx*dy)
        #print (point)

        transformedImage[y][x] = point

        #sys.stdin.readline()
      else:
        #print ("Not interpolating (%f, %f)" % (xOrig, yOrig))
        transformedImage[y][x] = originalArray[int(yOrig)][int(xOrig)]

  return transformedImage
