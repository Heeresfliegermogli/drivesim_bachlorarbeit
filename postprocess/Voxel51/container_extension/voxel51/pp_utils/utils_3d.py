import numpy as np

from .camera import project_fish_eye_polynomial, project_pinhole
from .utils import permute


def get_bbox_3d_corners(extents):
    """Return transformed points in the following order: [LDB, RDB, LUB, RUB, LDF, RDF, LUF, RUF]
    where R=Right, L=Left, D=Down, U=Up, B=Back, F=Front and LR: x-axis, UD: y-axis, FB: z-axis.

    Args:
        extents (numpy.ndarray): A structured numpy array containing the fields: [`x_min`, `y_min`,
            `x_max`, `y_max`, `transform`.

    Returns:
        (numpy.ndarray): Transformed corner coordinates with shape `(N, 8, 3)`.
    """

    rdb = [extents["x_max"], extents["y_min"], extents["z_min"]]
    ldb = [extents["x_min"], extents["y_min"], extents["z_min"]]
    lub = [extents["x_min"], extents["y_max"], extents["z_min"]]
    rub = [extents["x_max"], extents["y_max"], extents["z_min"]]
    ldf = [extents["x_min"], extents["y_min"], extents["z_max"]]
    rdf = [extents["x_max"], extents["y_min"], extents["z_max"]]
    luf = [extents["x_min"], extents["y_max"], extents["z_max"]]
    ruf = [extents["x_max"], extents["y_max"], extents["z_max"]]
    tfs = extents["transform"]

    corners = np.stack((ldb, rdb, lub, rub, ldf, rdf, luf, ruf), 0)
    corners_homo = np.pad(corners, ((0, 0), (0, 1), (0, 0)), constant_values=1.0)

    return np.einsum("jki,ikl->ijl", corners_homo, tfs)[..., :3]


def proj_3dbbox_to_image_coords(bbox3d_list, camera_params, im_size, im_center):
    r"""
    Takes as input a list of 3D bounding boxes in the world frame and projects them into 2D using the camera extrinsics.

    Args:
        bbox3d_list: A list of 3D bounding boxes in the world frame, each of which contain min / max extents for
                    x, y, z, and a 4x4 world transform.
        camera_params: Camera parameters that include a view transform matrix and a projection matrix.
        im_center: coordinates for the center of the image (H/2 and W/2)

    Returns:
        A list of 2D image coordinates for all 8 corners of the bounding box.

    """
    im_size_proj = list(im_size)

    data = {}  # noqa SIM904
    data["bounding_box_3d"] = {}
    data["bounding_box_3d"]["x_min"] = []
    data["bounding_box_3d"]["y_min"] = []
    data["bounding_box_3d"]["z_min"] = []
    data["bounding_box_3d"]["x_max"] = []
    data["bounding_box_3d"]["y_max"] = []
    data["bounding_box_3d"]["z_max"] = []
    data["bounding_box_3d"]["transform"] = []

    for bbox3d in bbox3d_list:
        data["bounding_box_3d"]["x_min"].append(bbox3d.xmin)
        data["bounding_box_3d"]["y_min"].append(bbox3d.ymin)
        data["bounding_box_3d"]["z_min"].append(bbox3d.zmin)
        data["bounding_box_3d"]["x_max"].append(bbox3d.xmax)
        data["bounding_box_3d"]["y_max"].append(bbox3d.ymax)
        data["bounding_box_3d"]["z_max"].append(bbox3d.zmax)
        data["bounding_box_3d"]["transform"].append(np.transpose(bbox3d.T))

    bbox3d_corners = get_bbox_3d_corners(data["bounding_box_3d"])
    bbox3d_proj = []

    for i, bbox3d in enumerate(bbox3d_list):  # noqa B007 - Skipping formatting since this is an adapted code
        points_in_world_frame = bbox3d_corners[i]
        if camera_params.cameraModel == "fisheyePolynomial" or camera_params.cameraModel == "ftheta":
            points_proj = project_fish_eye_polynomial(points_in_world_frame, camera_params)
            im_size_proj = [camera_params.cameraFisheyeNominalWidth, camera_params.cameraFisheyeNominalHeight]
            # print(points_in_world_frame, points_proj, im_size_proj, im_center, "points")
        else:
            points_proj = project_pinhole(points_in_world_frame, im_center, camera_params)

        points_proj = [
            (x / camera_params.cameraFisheyeNominalWidth, y / camera_params.cameraFisheyeNominalHeight)
            for x, y in points_proj
        ]
        points_proj = permute(points_proj, [5, 7, 3, 1, 4, 6, 2, 0])
        bbox3d_proj.append(list(points_proj))

    return bbox3d_proj, im_size_proj
