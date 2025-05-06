import numpy as np


class ObjectDetection:  # this can store both 2D and 3D bounding box data
    def __init__(self, object_type=None, npy_bbox2d=None, npy_bbox3d=None):
        if object_type is not None:
            self.type = object_type
        else:
            self.type = None
        # self.truncated = line[1]
        # self.occluded = line[2]
        # self.alpha = line[3]
        if npy_bbox2d is not None:
            self.xmin = npy_bbox2d[1]
            self.ymin = npy_bbox2d[2]
            self.xmax = npy_bbox2d[3]
            self.ymax = npy_bbox2d[4]
        # self.dimensions = line[8:11]
        # self.location = line[11:14]
        # self.rotation_y = line[14]
        # self.score = line[15]

        if npy_bbox3d is not None:
            # print(f"{self.type}, bbox3d: {npy_bbox3d}")
            self.xmin = npy_bbox3d[1]  # these mins and maxes specify the bbox extents of the canonical shape size.
            self.ymin = npy_bbox3d[2]
            self.zmin = npy_bbox3d[3]
            self.xmax = npy_bbox3d[4]
            self.ymax = npy_bbox3d[5]
            self.zmax = npy_bbox3d[6]
            self.T = np.transpose(npy_bbox3d[7])  # the diagonal elements scale the mins and maxes of the bbox extents


class CameraParams:  # this stores camera params, which are required for projecting a 3d bounding box into image space
    def __init__(self, camera_params):
        self.cameraFisheyeMaxFOV = camera_params["cameraFisheyeMaxFOV"]
        self.cameraFisheyeNominalHeight = camera_params["cameraFisheyeNominalHeight"]
        self.cameraFisheyeNominalWidth = camera_params["cameraFisheyeNominalWidth"]
        self.cameraFisheyeOpticalCentre = camera_params["cameraFisheyeOpticalCentre"]
        self.cameraFisheyePolynomial = camera_params["cameraFisheyePolynomial"]
        self.cameraModel = camera_params["cameraModel"]
        self.cameraNearFar = camera_params["cameraNearFar"]
        self.cameraProjection = np.transpose(np.array(camera_params["cameraProjection"]).reshape(4, 4))
        self.cameraViewTransform = np.transpose(np.array(camera_params["cameraViewTransform"]).reshape(4, 4))
        self.metersPerSceneUnit = camera_params["metersPerSceneUnit"]
