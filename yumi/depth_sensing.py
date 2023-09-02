########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    This sample demonstrates how to capture a live 3D point cloud   
    with the ZED SDK and display the result in an OpenGL window.    
"""

import sys
import ogl_viewer.viewer as gl
import pyzed.sl as sl
import numpy as np
import matplotlib.pyplot as plt

def parseArg(argLen, argv, param):
    if(argLen>1):
        if(".svo" in argv):
            # SVO input mode
            param.set_from_svo_file(sys.argv[1])
            print("Sample using SVO file input "+ sys.argv[1])
        elif(len(argv.split(":")) == 2 and len(argv.split(".")) == 4):
            #  Stream input mode - IP + port
            l = argv.split(".")
            ip_adress = l[0] + '.' + l[1] + '.' + l[2] + '.' + l[3].split(':')[0]
            port = int(l[3].split(':')[1])
            param.set_from_stream(ip_adress,port)
            print("Stream input mode")
        elif (len(argv.split(":")) != 2 and len(argv.split(".")) == 4):
            #  Stream input mode - IP
            param.set_from_stream(argv)
            print("Stream input mode")
        elif("HD2K" in argv):
            param.camera_resolution = sl.RESOLUTION.HD2K
            print("Using camera in HD2K mode")
        elif("HD1200" in argv):
            param.camera_resolution = sl.RESOLUTION.HD1200
            print("Using camera in HD1200 mode")
        elif("HD1080" in argv):
            param.camera_resolution = sl.RESOLUTION.HD1080
            print("Using camera in HD1080 mode")
        elif("HD720" in argv):
            param.camera_resolution = sl.RESOLUTION.HD720
            print("Using camera in HD720 mode")
        elif("SVGA" in argv):
            param.camera_resolution = sl.RESOLUTION.SVGA
            print("Using camera in SVGA mode")
        elif("VGA" in argv and "SVGA" not in argv):
            param.camera_resolution = sl.RESOLUTION.VGA
            print("Using camera in VGA mode")

def get_rgb_get_depth():
    print("Running Depth Sensing sample ... Press 'Esc' to quit\nPress 's' to save the point cloud")

    init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.ULTRA,
                                 coordinate_units=sl.UNIT.METER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
    if (len(sys.argv) > 1):
        parseArg(len(sys.argv), sys.argv[1], init)
    
    
    cam_id = 20120598 # cam id for the overhead
    # cam_id = 22008760 # cam id for the left camera
    
    init.set_from_serial_number(cam_id)
    
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    res = sl.Resolution()
    res.width = 720
    res.height = 404

    camera_model = zed.get_camera_information().camera_model
    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)
        # breakpoint()
    # Create OpenGL viewer
    # viewer = gl.GLViewer()
    # viewer.init(1, sys.argv, camera_model, res)

    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    depth = sl.Mat()
    left = sl.Mat()

    while True:
        if zed.is_opened():
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)
                zed.retrieve_image(left, sl.VIEW.LEFT)
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                break
    zed.close()

    depth_img = depth.get_data()
    # REMINDER : 12.7mm for half an inch!!!!
    # set values in the image that are nan to 0 so we can display it 
    depth_img = np.where(np.isnan(depth_img) | np.isinf(depth_img), 0, depth_img)
    depth_img[np.isnan(depth_img)]=0
    plt.imshow(depth_img, interpolation='nearest')
    plt.show()
    print("depth img shape: ", depth_img.shape)

    rgb_img = left.get_data()
    # plt.imshow(rgb_img)
    # plt.show()


    # display results
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                            sharex=True, sharey=True)

    ax = axes.ravel()

    ax[0].imshow(depth_img, interpolation='nearest')
    ax[0].axis('off')
    ax[0].set_title('depth_img', fontsize=20)

    ax[1].imshow(rgb_img)
    ax[1].axis('off')
    ax[1].set_title('rgb_img', fontsize=20)

    fig.tight_layout()
    plt.show()


    # print("rgb img shape: ", rgb_img.shape)
    print("depth img type: ", depth_img.dtype)

    return depth_img, rgb_img

if __name__ == "__main__":
    get_rgb_get_depth()
