import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time 

def setup_rs_camera(use_depth=False):
    config = rs.config()
    pipeline = rs.pipeline()

    # Disable continuous stream loop
    config.enable_stream(rs.stream.color)
    if use_depth:
        config.enable_stream(rs.stream.depth)

    profile = pipeline.start(config)
    colorizer = rs.colorizer()

    align_to = rs.stream.color
    align = rs.align(align_to)

    if use_depth:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
    else:
        depth_scale = None

    return pipeline, colorizer, align, depth_scale

def get_rs_image(pipeline, align, depth_scale, use_depth=False):
    frames = pipeline.wait_for_frames(timeout_ms=5000)

    # align the deph to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    if use_depth:
        aligned_depth_frame = aligned_frames.get_depth_frame()
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        scaled_depth_image = depth_image * depth_scale
    else:
        aligned_depth_frame = None
        scaled_depth_image = None

    color_frame = aligned_frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    return color_image, scaled_depth_image, aligned_depth_frame

def viz_rs_images(color_image, colorizer, aligned_depth_frame, x_crop, y_crop, use_depth=False):
    # convert color image to BGR for OpenCV
    r, g, b = cv2.split(color_image)
    color_image = cv2.merge((b, g, r))
    cv2.imshow('Cropped Color Image',
                        color_image[x_crop[0]:x_crop[1] , y_crop[0]:y_crop[1]])
    if use_depth:
        depth_colormap = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        cv2.imshow('Cropped Depth Image',
                            depth_colormap[x_crop[0]:x_crop[1] , y_crop[0]:y_crop[1]])
    cv2.waitKey()
    return None

def main():
    use_depth = True
    pipeline, colorizer, align, depth_scale = setup_rs_camera(use_depth=use_depth)
    time.sleep(1)
    color_image, scaled_depth_image, aligned_depth_frame = get_rs_image(
                                pipeline, align, depth_scale, use_depth=use_depth)
    # Values to crop images to fit workspace
    y_crop = [0, 1280]
    x_crop = [0, 720]
    plt.imshow(color_image)
    plt.show()
    cv2.imshow('Image', color_image)
    cv2.waitKey()
    viz_rs_images(color_image, colorizer, aligned_depth_frame, x_crop, y_crop, use_depth=use_depth)

if __name__ == "__main__":
    main()
