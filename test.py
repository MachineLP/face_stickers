import cv2
import time
from numba import njit
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from numba import jit, prange


# 仿射变换，用于将贴图点映射到人脸点
def get_text_trans_matrix(x1, y1, x2, y2, x3, y3, tx1, ty1, tx2, ty2, tx3, ty3):
    # 放射变换
    return cv2.getAffineTransform( np.float32([ [tx1, ty1], [tx2, ty2], [tx3, ty3] ]), np.float32( [ [x1, y1], [x2, y2], [x3, y3] ]) ).flatten()
    # 透视变换
    # return cv2.getPerspectiveTransform( np.float32([ [tx1, ty1], [tx2, ty2], [tx3, ty3] ]), np.float32( [ [x1, y1], [x2, y2], [x3, y3] ]) ).flatten()

@jit(nopython=True)
def sticker(srcData, width, height, stride, mask, maskWidth, maskHeight, maskStride, srcFacePoints, maskFacePoints, H):
    def CLIP3(x, a, b):
        return min(max(a,x), b)
    # 用于将贴图点映射到人脸点 
    for i in range(height):
    # for i in prange(height):
        for j in range(width):
            x = float(i)
            y = float(j)
            tx = (int)((H[0] * (x)+H[1] * (y)+H[2]) + 0.5)
            ty = (int)((H[3] * (x)+H[4] * (y)+H[5]) + 0.5)
            tx = CLIP3(tx, 0, maskHeight - 1)
            ty = CLIP3(ty, 0, maskWidth - 1)	
            mr = int( mask[ int(tx), int(ty), 0 ] ) 
            mg = int( mask[ int(tx), int(ty), 1 ] ) 
            mb = int( mask[ int(tx), int(ty), 2 ] ) 
            alpha = int( mask[ int(tx), int(ty), 3 ] )  
            #if alpha!=0:
            #    print( '>>>', alpha )
            b = srcData[i, j, 0]
            g = srcData[i, j, 1]
            r = srcData[i, j, 2]		
            srcData[i, j, 0] =CLIP3((b * (255 - alpha) + mb * alpha) / 255, 0, 255)
            srcData[i, j, 1] =CLIP3((g * (255 - alpha) + mg * alpha) / 255, 0, 255)
            srcData[i, j, 2] =CLIP3((r * (255 - alpha) + mr * alpha) / 255, 0, 255)
    return srcData


# @jit(parallel=True,nogil=True)
# @njit(parallel=True,nogil=True)
def trent_sticker(srcData, width, height, stride, mask, maskWidth, maskHeight, maskStride, srcFacePoints, maskFacePoints, ratio):
    ret = 0
    H = get_text_trans_matrix( maskFacePoints[0], maskFacePoints[1],maskFacePoints[2],maskFacePoints[3],maskFacePoints[4],maskFacePoints[5], srcFacePoints[0], srcFacePoints[1],srcFacePoints[2],srcFacePoints[3],srcFacePoints[4],srcFacePoints[5] )
    #print ('H', H) 
    srcData = sticker(srcData, width, height, stride, mask, maskWidth, maskHeight, maskStride, srcFacePoints, maskFacePoints, H)
    return srcData, ret 



img = Image.open('./mask_img/mask_b.png') 
r,g,b,a=img.split()   #分离4通道 
print (r,g,b,a) 
im_array = np.array(img) 
# 高宽通道    (坐标取值为y, x) 
mask_h, mask_w, mask_c = im_array.shape 
print ('>>>>>>>', mask_h, mask_w, mask_c) 


mtcnn = MTCNN('./pb/mtcnn.pb')
img = cv2.imread( './test_img/004.jpg' )
# 高宽通道    (坐标取值为y, x)
h, w, c = img.shape
print ('>>>>>>>', h, w, c) 
bbox, scores, landmarks = mtcnn.detect(img)
print ('>>>>>', len( bbox) )
for box, pts in zip(bbox, landmarks):
    faceInfos = np.array( [ 1, box[1], box[0], box[3] - box[1], box[2] - box[0], pts[5], pts[0], pts[6], pts[1], pts[7], pts[2], pts[8], pts[3], pts[9], pts[4] ] )


# 左眼
# 右眼
# 两嘴角中间
# 坐标为 x, y
# srcFacePoints = np.array( [faceInfos[5], faceInfos[6], faceInfos[7], faceInfos[8], (faceInfos[11] + faceInfos[13])/2.0, (faceInfos[12]+faceInfos[14])/2.0 ]) 
# 坐标转为 y, x
srcFacePoints = np.array( [faceInfos[6], faceInfos[5], faceInfos[8], faceInfos[7], (faceInfos[12]+faceInfos[14])/2.0, (faceInfos[11] + faceInfos[13])/2.0 ] ) 
print ('srcFacePoints:', srcFacePoints) 

# 307, 364, 423, 364, 365, 490
# 左眼
# 右眼
# 两嘴角中间
# 坐标为 x, y
# maskFacePoints = np.array( [ 307.0, 364.0, 423.0, 364.0, 365.0, 490.0 ] ) 
# 坐标为 y, x
maskFacePoints = np.array( [ 364.0, 307.0,  364.0, 423.0, 490.0 , 365.0 ] ) 
print ('maskFacePoints:', maskFacePoints)


##### trent_sticker(srcData, width, height, stride, mask, maskWidth, maskHeight, maskStride, srcFacePoints, maskFacePoints, ratio) 
start_time = time.time()
srcData, ret  = trent_sticker( img, w, h, 3, im_array, mask_w, mask_h, 4, srcFacePoints, maskFacePoints, 100 ) 
print ( 'time >>>>', time.time() - start_time )
img_mask = np.array(srcData, dtype=np.uint8) 
cv2.imwrite('res.jpg', img_mask) 





#global pts
#global box
pts = [0,0,0,0,0,0,0,0,0,0]
box = [0,0,0,0]
def interface(img):
    global pts, box
    h, w, c = img.shape
    bbox, scores, landmarks = mtcnn.detect(img)

    try:
        # for box, pts in zip(bbox, landmarks):
        # box = bbox[0]; pts = landmarks[0]
        if abs( bbox[0][0] - box[0])  > 7 or abs( bbox[0][1] - box[1]) > 7 or abs( bbox[0][2] - box[2]) > 7 or abs( bbox[0][3] - box[3]) > 7:
            box = bbox[0]; pts = landmarks[0]
        faceInfos_t = np.array( [ 1, box[1], box[0], box[3] - box[1], box[2] - box[0], pts[5], pts[0], pts[6], pts[1], pts[7], pts[2], pts[8], pts[3], pts[9], pts[4] ] )
        srcFacePoints = np.array( [faceInfos_t[6], faceInfos_t[5], faceInfos_t[8], faceInfos_t[7], (faceInfos_t[12]+faceInfos_t[14])/2.0, (faceInfos_t[11] + faceInfos_t[13])/2.0 ] ) 
        srcData, ret  = trent_sticker( img, w, h, 3, im_array, mask_w, mask_h, 4, srcFacePoints, maskFacePoints, 100 )
    except:
        srcData = img
    
    return srcData


