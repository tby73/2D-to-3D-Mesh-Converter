import cv2
import torch
import open3d as o3d 
import numpy as np
import matplotlib.pyplot as plt

from transformers import GLPNImageProcessor, GLPNForDepthEstimation

def DisplayImage(input, colormap):
    plt.imshow(input, cmap=colormap)
    plt.show()

def GenerateDepthMap(input_image):
    # load image processing and depth estimation models
    feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

    inputs = feature_extractor(images=input_image, return_tensors="pt")

    # get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # process output into optimal sizes
    padding = 16
    output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    output = output[padding:-padding, padding:-padding]

    return output

def GeneratePointcloud3D(input_image, depth_map):
    width, height = input_image.shape[1], input_image.shape[0]
    input_image = np.array(input_image)

    # process depth map
    depth_map = (depth_map * 255 / np.max(depth_map)).astype("uint8")

    # create RGBD image
    depth_3d = o3d.geometry.Image(depth_map)
    image_3d = o3d.geometry.Image(input_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_3d, depth_3d, convert_rgb_to_density=False)

    # camera settings 
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsic(width, height, 500, 500, width / 2, height / 2)

    # generate pointcloud 
    pointcloud = o3d.geometry.PointCloud.creaet_from_rgbd_image(rgbd_image, camera_intrinsic)
    o3d.visualization.draw_geometries([pointcloud])

def main():
    # input image path
    INPUT_PATH = "src/Images/sample1.jpg"

    # load input image
    input_image = cv2.imread(INPUT_PATH)
    input_image = cv2.resize(input_image, (1000, 1000), interpolation=cv2.INTER_LINEAR)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # get depth map
    depth_map = GenerateDepthMap(input_image)
    DisplayImage(depth_map, "plasma")

    # get Pointcloud
    GeneratePointcloud3D(input_image, depth_map)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()
