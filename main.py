import cv2
import numpy
from matplotlib import pyplot as plt
import open3d as o3d

#Input stereo image pair
img0 = cv2.imread("im0.png")
img1 = cv2.imread("im1.png")

#Specify camera intrinsics
focal = float(img0.shape[1]/2)
T = 150.
K = numpy.array((
    [focal,0,img0.shape[1]/2],
    [0,focal,img0.shape[0]/2],
    [0,0,1]),dtype=float)

#Convert stereo images into grayscale
img0_gray = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

#Compute disparity map with stereo matching(block matching)
stereo = cv2.StereoBM.create(numDisparities=64, blockSize=7)
disparity = (stereo.compute(img0_gray,img1_gray).astype(numpy.float32)+0.0001)/16. #disparity values were multiplied by 16 from real disparity
plt.imshow(disparity)
plt.show()

depth = T * focal/disparity #depth is obtained from T*F/disparity, assuming T = 100
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=o3d.geometry.Image(img0), depth=o3d.geometry.Image(depth))

#Create pointsCloud from RGBD Image
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(width=img0.shape[1], height=img0.shape[0],intrinsic_matrix=K))

vis = o3d.visualization
currentPose = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]], dtype=numpy.float64)
vis.draw(pcd, width=img0.shape[1], height=img0.shape[0], intrinsic_matrix=K, extrinsic_matrix=currentPose)