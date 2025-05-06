"""
Adapted from omni.replicator.insight
"""


import numpy as np


def ftheta_distortion(ftheta, x):
    """F-Theta distortion."""
    return ftheta["poly_a"] + x * (
        ftheta["poly_b"] + x * (ftheta["poly_c"] + x * (ftheta["poly_d"] + x * ftheta["poly_e"]))
    )


def project_fish_eye_map_to_sphere(direction):
    z = direction[:, 2:]
    cos_theta = -z
    theta = np.arccos(np.clip(cos_theta, 0.0, 1.0))
    theta = np.arccos(cos_theta)

    # TODO currently projecting outside of max FOV
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta + 1e-8)
    xy = direction[:, :2] / (sin_theta + 1e-8)
    return xy, theta


def ftheta_distortion_prime(ftheta, x):
    """Derivative to f_theta_distortion."""
    return ftheta["poly_b"] + x * (2 * ftheta["poly_c"] + x * (3 * ftheta["poly_d"] + x * 4 * ftheta["poly_e"]))


def _ftheta_distortion_solver(ftheta, y):
    # Guess by linear approximation. 2 loops provides sufficient precision in working range.
    ratio = ftheta["width"] / 2 / ftheta["edge_fov"]
    guess = y * ratio
    for _ in range(2):
        guessed_y = ftheta_distortion(ftheta, guess)
        dy = y - guessed_y
        dx = ftheta_distortion_prime(ftheta, guess)
        mask = dx != 0
        guess[mask] += dy[mask] / dx[mask]
        guess[~mask] += dy[~mask] * ratio
    return guess


# should go: +right, up/down, -deep
# currently goes: deep, left, up/down
def project_fish_eye_polynomial(points, view_params_obj):
    """Project F-Theta camera model.

    Args:
        points (numpy.ndarray): Array of points in world frame of shape (num_points, 3).
        view_params (dict): dictionary containing view parameters

    Returns:
        (numpy.ndarray): Image-space points of shape (num_points, 3)
    """
    ftheta = {}  # noqa SIM904
    ftheta["width"] = view_params_obj.cameraFisheyeNominalWidth
    ftheta["height"] = view_params_obj.cameraFisheyeNominalHeight
    ftheta["cx"] = view_params_obj.cameraFisheyeOpticalCentre[0]
    ftheta["cy"] = view_params_obj.cameraFisheyeOpticalCentre[1]
    ftheta["max_fov"] = view_params_obj.cameraFisheyeMaxFOV
    ftheta["poly_a"] = view_params_obj.cameraFisheyePolynomial[0]
    ftheta["poly_b"] = view_params_obj.cameraFisheyePolynomial[1]
    ftheta["poly_c"] = view_params_obj.cameraFisheyePolynomial[2]
    ftheta["poly_d"] = view_params_obj.cameraFisheyePolynomial[3]
    ftheta["poly_e"] = view_params_obj.cameraFisheyePolynomial[4]

    ftheta["edge_fov"] = ftheta_distortion(ftheta, ftheta["width"] / 2)
    ftheta["c_ndc"] = np.array(
        [
            (ftheta["cx"] - ftheta["width"] / 2) / ftheta["width"],
            (ftheta["height"] / 2 - ftheta["cy"]) / ftheta["width"],
        ]
    )

    view_params = {}  # noqa SIM904
    view_params["ftheta"] = ftheta
    view_params["world_to_view"] = view_params_obj.cameraViewTransform
    view_params["world_to_view"] = np.transpose(view_params["world_to_view"])

    points_h = np.pad(points, ((0, 0), (0, 1)), constant_values=1)
    points_cam_frame = np.einsum("jk,kl->jl", points_h, view_params["world_to_view"])[..., :3]
    # [  3.0184363   -1.40037273 -18.13704   ]]
    # points_cam_frame = np.concatenate((-points_cam_frame[:, 1:2], points_cam_frame[:, 2:3],
    # -points_cam_frame[:, 0:1]), axis = 1)
    # points_cam_frame = np.pad(points, ((0, 0), (0, 1)), constant_values=1)
    # print(points_cam_frame, "POINTS CAM FRAME")

    directions = points_cam_frame / np.linalg.norm(points_cam_frame + 1e-8, axis=1)[:, None]
    xy, theta = project_fish_eye_map_to_sphere(directions)
    r = _ftheta_distortion_solver(view_params["ftheta"], theta)
    screen = xy * r
    screen = screen / view_params["ftheta"]["width"] + view_params["ftheta"]["c_ndc"]
    screen[:, 1] *= -(view_params["ftheta"]["width"] / view_params["ftheta"]["height"])
    ndc = screen + 0.5

    ndc[:, 0] *= ftheta["width"]
    ndc[:, 1] *= ftheta["height"]
    # ndc = np.pad(ndc, ((0, 0), (0, 1)), constant_values=0)
    return ndc


def project_pinhole(points, im_center, view_params_obj):
    """

    Project 3D points to 2D camera view using a pinhole camera model.

    Args:
        points (numpy.ndarray): Array of points in world frame of shape (num_points, 3).
        viewport (omni.kit.viewport_legacy._viewport.IViewportWindow): Viewport from which to retrieve/create sensor.

    Returns:
        (numpy.ndarray): Image-space points of shape (num_points, 3)
    """

    view_params = {}
    view_params["camera_projection"] = view_params_obj.cameraProjection
    view_params["aspect_ratio"] = view_params_obj.cameraProjection[0][0] / view_params_obj.cameraProjection[1][1]
    view_params["z_near"] = 0.1
    view_params["z_far"] = 1000000
    view_params["world_to_view"] = view_params_obj.cameraViewTransform
    view_params["world_to_view"] = np.transpose(view_params["world_to_view"])

    view_proj_matrix = np.dot(view_params["world_to_view"], view_params["camera_projection"])
    homo = np.pad(points, ((0, 0), (0, 1)), constant_values=1.0)
    tf_points = np.dot(homo, view_proj_matrix)
    tf_points = tf_points / (tf_points[..., -1:])
    tf_points[..., :2] = 0.5 * (tf_points[..., :2] + 1)
    tf_points[:, 0] = 1 - tf_points[:, 0]
    tf_points[:, 0] *= im_center[0] * 2
    tf_points[:, 1] *= im_center[1] * 2

    return tf_points[..., :3]
