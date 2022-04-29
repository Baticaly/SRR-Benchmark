# Benchmark Process Class

import logging
from operator import index
from textwrap import indent
import time
import os
import cv2
import numpy as np
import imutils

class Processes:
    def sequentialScheme(threadLock, inputList):
        threadLock.acquire()
        logging.info('Keypoint Selector flag')
        logging.info('Process ID %s \n', os.getpid())

        orb = cv2.ORB_create()

        imageStack, initialSource, initialKeypoint, initialDescriptor = None, None, None, None

        for input in inputList:
            source = cv2.imread(input)
            source_float = source.astype(np.float32)

            keypoints, descriptors = orb.detectAndCompute(source,None)

            # Individual detect & compute
            # keypoints = orb.detect(source, None)
            # keypoints, descriptors = orb.compute(source, keypoints)

            BFmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

            if initialSource is not None: 
                matchList = BFmatcher.match(initialDescriptor, descriptors)
                matchList = sorted(matchList, key=lambda x: x.distance)

                sourcePoints  = np.float32([initialKeypoint[m.queryIdx].pt for m in matchList]).reshape(-1,1,2)
                destinationPoints  = np.float32([keypoints[m.trainIdx].pt for m in matchList]).reshape(-1,1,2)

                M, mask = cv2.findHomography(sourcePoints, destinationPoints, cv2.RANSAC, 5.0)
                w, h, _ = source_float.shape
                source_float = cv2.warpPerspective(source_float, M, (h, w))
                imageStack += source_float

                # beta = (1.0 - alpha)
                # dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)

            else:
                imageStack = source_float
                initialSource = source
                initialKeypoint = keypoints
                initialDescriptor = descriptors

        imageStack /= len(inputList)
        imageStack = (imageStack).astype(np.uint8)
        cv2.imwrite('images/result.jpg', imageStack)

        threadLock.release()

    def pyramidScheme(threadLock, inputList):

        '''
        Only first row done, will add following rows
        '''

        threadLock.acquire()

        orb = cv2.ORB_create()

        lastIndex = 0
        stack, source, keypoints, descriptor = None, None, None, None

        for imageIndex in range(0, len(inputList) -1):
            source1 = cv2.imread(inputList[imageIndex])
            source2 = cv2.imread(inputList[imageIndex + 1])
                
            source1_float = source1.astype(np.float32)
            source2_float = source2.astype(np.float32)

            s1keypoints, s1descriptors = orb.detectAndCompute(source1,None)
            s2keypoints, s2descriptors = orb.detectAndCompute(source2,None)

            BFmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

            matchList = BFmatcher.match(s1descriptors, s2descriptors)
            matchList = sorted(matchList, key=lambda x: x.distance)

            sourcePoints  = np.float32([s1keypoints[m.queryIdx].pt for m in matchList]).reshape(-1,1,2)
            destinationPoints  = np.float32([s2keypoints[m.trainIdx].pt for m in matchList]).reshape(-1,1,2)

            M, mask = cv2.findHomography(sourcePoints, destinationPoints, cv2.RANSAC, 5.0)
            w, h, _ = source2_float.shape
            source2_float = cv2.warpPerspective(source2_float, M, (h, w))
            result = cv2.addWeighted(source1_float, 0.5, source2_float, 0.5 , 0.0)

            result = (result).astype(np.uint8)
            cv2.imwrite('images/result.jpg', result)

        threadLock.release()

    def interpolationDemo(threadLock, inputList):
        threadLock.acquire()

        logging.info('stitchDemo Selector flag')
        logging.info('Process ID %s \n', os.getpid())

        '''
        OpenCV image read, orb initialization
        '''

        cv2.ocl.setUseOpenCL(False)

        # orb = cv2.ORB_create()
        orb = cv2.ORB_create(nfeatures=500000, edgeThreshold=10)

        query = cv2.imread(inputList[0])        #Base Image
        train = cv2.imread(inputList[1])

        query_float = query.astype(np.float32)
        train_float = train.astype(np.float32)

        '''
        ORB Keypoint & Feature compute
        '''

        (kpsA, featuresA) = orb.detectAndCompute(train, None)
        (kpsB, featuresB) = orb.detectAndCompute(query, None)
        
        '''
        Norm-Hamming brute-force match
        '''

        BFmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

        matchList = BFmatcher.match(featuresA, featuresB)
        matchList = sorted(matchList, key=lambda x: x.distance)

        # Match Display
        matchImage = cv2.drawMatches(train,kpsA,query,kpsB,matchList[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("matches", matchImage)
        cv2.waitKey();cv2.destroyAllWindows()

        kpsA = np.float32([kp.pt for kp in kpsA])
        kpsB = np.float32([kp.pt for kp in kpsB])

        ptsA = np.float32([kpsA[m.queryIdx] for m in matchList])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matchList])

        '''
        OpenCV homography compute & perspective warp
        '''

        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 1)

        width = train.shape[1] + query.shape[1]
        height = train.shape[0] + query.shape[0]

        result = cv2.warpPerspective(train_float, H, (width, height))

        '''
        Stacking & Interpolation
        '''

        # Basic Stacking
        # width, height, _ = query.shape
        # result[0:height, 0:width] = query

        # Interpolation
        # width, height, _ = query.shape
        # for h in range(0, height):
        #     for w in range(0, width):
        #         result[h][w] = result[h][w] + query_float[h][w]

        # result = result / 2
        # result = result.astype(np.uint8)

        # Interpolation & brightness adjustment (Common keypoint evaluation)
        width, height, _ = query.shape
        for h in range(0, height):
            for w in range(0, width):
                resultValue = result[h][w][0] + result[h][w][1] + result[h][w][2]
                queryValue = query_float[h][w][0] + query_float[h][w][1] + query_float[h][w][2]
                if queryValue != 0 and resultValue != 0:
                    result[h][w] = ( result[h][w] + query_float[h][w] ) / 2
                else:
                    result[h][w] = result[h][w] + query_float[h][w]

        result = result.astype(np.uint8)

        '''
        Post-process
        '''

        # Border remove
        mask = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        threshold = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]

        contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        c = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        result = result[y:y + h, x:x + w]

        cv2.imwrite('result.jpg', result)

        threadLock.release()

    def progressCache(threadLock, inputList):
        threadLock.acquire()
        threadLock.release()