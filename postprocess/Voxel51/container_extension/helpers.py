import math

import numpy as np

EPS = 1e-8

# from pdb import set_trace


def get_sensor_name(data):
    return data["render_products"]["camera"].split("/")[-1]


def get_view_params(data, render_product_name="render_products"):
    camera_params = data["camera_params"]

    world_to_view = np.array(camera_params["cameraViewTransform"]).reshape(4, 4)
    width, height = data[render_product_name]["resolution"]

    projection_type = camera_params["cameraModel"]

    if projection_type == "fisheyePolynomial":
        ftheta = {
            "width": camera_params["cameraFisheyeNominalWidth"],
            "height": camera_params["cameraFisheyeNominalHeight"],
            "cx": camera_params["cameraFisheyeOpticalCentre"][0],
            "cy": camera_params["cameraFisheyeOpticalCentre"][1],
            "poly_a": camera_params["cameraFisheyePolynomial"][0],
            "poly_b": camera_params["cameraFisheyePolynomial"][1],
            "poly_c": camera_params["cameraFisheyePolynomial"][2],
            "poly_d": camera_params["cameraFisheyePolynomial"][3],
            "poly_e": camera_params["cameraFisheyePolynomial"][4],
            "max_fov": camera_params["cameraFisheyeMaxFOV"],
        }
        ftheta["edge_fov"] = ftheta_distortion(ftheta, ftheta["width"] / 2)
        ftheta["c_ndc"] = np.array(
            [
                (ftheta["cx"] - ftheta["width"] / 2) / ftheta["width"],
                (ftheta["height"] / 2 - ftheta["cy"]) / ftheta["width"],
            ]
        )
    else:
        ftheta = None

    return {
        "view_to_world": np.linalg.inv(np.array(world_to_view)),
        "world_to_view": np.array(world_to_view),
        "projection_type": projection_type,
        "ftheta": ftheta,
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "clipping_range": camera_params["cameraNearFar"],
        "horizontal_aperture": 0,  # TODO jlafleche: Obtain aperture from projection matrix
        "focal_length": 0,  # TODO jlafleche: Obtain aperture from projection matrix,
    }


def _interpolate(p, a, b):
    p0 = 1.0 - p
    return [int(p0 * a[0] + p * b[0]), int(p0 * a[1] + p * b[1]), int(p0 * a[2] + p * b[2]), 255]


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


def reduce_bboxes_2d(bboxes, instance_mappings):
    """
    Reduce 2D bounding boxes of leaf nodes to prims with a semantic label.

    Args:
        bboxes (numpy.ndarray): A structured numpy array containing the fields: `[
            ("instanceId", "<u4"), ("semanticId", "<u4"), ("x_min", "<i4"),
            ("y_min", "<i4"), ("x_max", "<i4"), ("y_max", "<i4")]`
        instance_mappings (numpy.ndarray): A structured numpy array containing the fields:
            `[("uniqueId", np.int32), ("name", "O"), ("semanticId", "<u4"), ("semanticLabel", "O"),
              ("instanceIds", "O"), ("metadata", "O")]`

    Returns:
        (numpy.ndarray): A structured numpy array containing the fields: `[
            ("uniqueId", np.int32), ("name", "O"), ("semanticLabel", "O"), ("instanceIds", "O"),
            ("semanticId", "<u4"), ("metadata", "O"), ("x_min", "<i4"), ("y_min", "<i4"),
            ("x_max", "<i4"), ("y_max", "<i4")]`
    """
    bboxes = bboxes[bboxes["x_min"] < 2147483647]
    reduced_bboxes = []
    for im in instance_mappings:
        if im["instanceIds"]:  # if mapping has descendant instance ids
            mask = np.isin(bboxes["instanceId"], im["instanceIds"])
            bbox_masked = bboxes[mask]
            if len(bbox_masked) > 0:
                reduced_bboxes.append(
                    (
                        im["uniqueId"],
                        im["name"],
                        im["semanticLabel"],
                        im["metadata"],
                        im["instanceIds"],
                        im["semanticId"],
                        np.min(bbox_masked["x_min"]),
                        np.min(bbox_masked["y_min"]),
                        np.max(bbox_masked["x_max"]),
                        np.max(bbox_masked["y_max"]),
                    )
                )
    return np.array(
        reduced_bboxes,
        dtype=[("uniqueId", np.int32), ("name", "O"), ("semanticLabel", "O"), ("metadata", "O"), ("instanceIds", "O")]
        + bboxes.dtype.descr[1:],
    )


def get_projection_matrix(fov, aspect_ratio, z_near, z_far):
    """
    Calculate the camera projection matrix.

    Args:
        fov (float): Field of View (in radians)
        aspect_ratio (float): Image aspect ratio (Width / Height)
        z_near (float): distance to near clipping plane
        z_far (float): distance to far clipping plane

    Returns:
        (numpy.ndarray): View projection matrix with shape `(4, 4)`
    """
    a = -1.0 / math.tan(fov / 2)
    b = -a * aspect_ratio
    c = z_far / (z_far - z_near)
    d = z_near * z_far / (z_far - z_near)
    return np.array([[a, 0.0, 0.0, 0.0], [0.0, b, 0.0, 0.0], [0.0, 0.0, c, 1.0], [0.0, 0.0, d, 0.0]])


def get_view_proj_mat(view_params):
    """
    Get View Projection Matrix.

    Args:
        view_params (dict): dictionary containing view parameters
    """
    z_near, z_far = view_params["clipping_range"]
    view_matrix = view_params["world_to_view"]
    fov = 2 * math.atan(view_params["horizontal_aperture"] / (2 * view_params["focal_length"]))
    projection_mat = get_projection_matrix(fov, view_params["aspect_ratio"], z_near, z_far)
    return np.dot(view_matrix, projection_mat)


def reduce_occlusion(occlusion_data, instance_mappings):
    """
    Reduce occlusion value of leaf nodes to prims with a semantic label.

    Args:
        sensor_data (numpy.ndarray): A structured numpy array with the fields: [("instanceId", "<u4"),
            ("semanticId", "<u4"), ("occlusionRatio", "<f4")], where occlusion ranges from 0
            (not occluded) to 1 (fully occluded).

    Returns:
        (numpy.ndarray): A structured numpy array with the fields: [("uniqueId", np.int32)
            ("name", "O"), ("semanticLabel", "O"), ("instanceIds", "O"), ("semanticId", "<u4"),
            ("metadata", "O"), ("occlusionRatio", "<f4")]
    """
    mapped_data = []
    occlusion_data = occlusion_data[~np.isnan(occlusion_data["occlusionRatio"])]
    for im in instance_mappings:
        if im["instanceIds"]:  # if mapping has descendant instance ids
            mask = np.isin(occlusion_data["instanceId"], im["instanceIds"])
            if mask.sum() > 1:
                print(
                    f"""[syntheticdata.viz] Mapping on {im['name']} contains multiple child meshes,
                      occlusion value may be incorrect."""
                )
            occ = occlusion_data[mask]
            if len(occ) > 0:
                mapped_data.append(
                    (
                        im["uniqueId"],
                        im["name"],
                        im["semanticLabel"],
                        im["metadata"],
                        im["instanceIds"],
                        im["semanticId"],
                        np.mean(occ["occlusionRatio"]),
                    )
                )
    return np.array(
        mapped_data,
        dtype=[("uniqueId", np.int32), ("name", "O"), ("semanticLabel", "O"), ("metadata", "O"), ("instanceIds", "O")]
        + occlusion_data.dtype.descr[1:],
    )


def _join_struct_arrays(arrays):
    """
    Join N numpy structured arrays.
    """
    n = len(arrays[0])
    assert all([len(a) == n for a in arrays])  # noqa C419: Possibly unused and could be deleted
    dtypes = sum(([d for d in a.dtype.descr if d[0]] for a in arrays), [])
    joined = np.empty(n, dtype=dtypes)
    for a in arrays:
        joined[list(a.dtype.names)] = a
    return joined


def _fish_eye_map_to_sphere(screen, screen_norm, theta, max_fov):
    """Utility function to map a sample from a disk on the image plane to a sphere."""
    direction = np.array([[0, 0, -1]] * screen.shape[0], dtype=np.float)
    extent = np.zeros(screen.shape[0], dtype=np.float)
    # A real fisheye have some maximum FOV after which the lens clips.
    valid_mask = theta <= max_fov
    # Map to a disk: screen / R normalizes the polar direction in screen space.
    xy = screen[valid_mask]
    screen_norm_mask = screen_norm[valid_mask] > 1e-5
    xy[screen_norm_mask] = xy[screen_norm_mask] / screen_norm[valid_mask, None]

    # Map disk to a sphere
    cos_theta = np.cos(theta[valid_mask])
    sin_theta = np.sqrt(1.0 - cos_theta**2)

    # Todo: is this right? Do we assume z is negative (RH coordinate system)?
    z = -cos_theta
    xy = xy * sin_theta[:, None]
    direction[valid_mask] = np.stack([xy[valid_mask, 0], xy[valid_mask, 1], z], axis=1)
    extent[valid_mask] = 1.0  # < far clip is not a plane, it's a sphere!

    return direction, extent


def project_fish_eye_map_to_sphere(direction):
    z = direction[:, 2:]
    cos_theta = -z
    theta = np.arccos(cos_theta)

    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta + EPS)
    xy = direction[:, :2] / (sin_theta + EPS)
    return xy, theta


def fish_eye_polynomial(ndc, view_params):
    """FTheta camera model based on DW src/rigconfiguration/CameraModelsNoEigen.hpp"""

    # Convert NDC pixel position to screen space... well almost. It is screen space but the extent of x is [-0.5, 0.5]
    # and the extent of y is [-0.5/aspectRatio, 0.5/aspectRatio].
    screen = ndc - 0.5
    aspect_ratio = view_params["width"] / view_params["height"]
    screen[:, 1] /= -aspect_ratio

    # The FTheta polynomial works at a nominal resolution. So far we have done calculations in NDC to be
    # resolution independent. Here we scale by the nominal resolution in X.
    screen = (screen - view_params["ftheta"]["c_ndc"]) * view_params["ftheta"]["width"]

    # Compute the radial distance on the screen from its center point
    r = np.sqrt(screen[:, 0] ** 2 + screen[:, 1] ** 2)
    theta = ftheta_distortion(view_params["ftheta"], r)
    max_fov = math.radians(view_params["ftheta"]["max_fov"] / 2)
    return _fish_eye_map_to_sphere(screen, r, theta, max_fov)


def project_fish_eye_polynomial(points, view_params):
    """Project F-Theta camera model.

    Args:
        points (numpy.ndarray): Array of points in world frame of shape (num_points, 3).
        view_params (dict): dictionary containing view parameters

    Returns:
        (numpy.ndarray): Image-space points of shape (num_points, 3) where the last component is 0.
    """
    points_h = np.pad(points, ((0, 0), (0, 1)), constant_values=1)
    points_cam_frame = np.einsum("jk,kl->jl", points_h, view_params["world_to_view"])[..., :3]
    directions = points_cam_frame / np.linalg.norm(points_cam_frame + EPS, axis=1)[:, None]
    xy, theta = project_fish_eye_map_to_sphere(directions)
    r = _ftheta_distortion_solver(view_params["ftheta"], theta)
    screen = xy * r
    screen = screen / view_params["ftheta"]["width"] + view_params["ftheta"]["c_ndc"]
    screen[:, 1] *= -(view_params["ftheta"]["width"] / view_params["ftheta"]["height"])
    ndc = screen + 0.5

    return np.pad(ndc, ((0, 0), (0, 1)), constant_values=0), True


def image_to_world(image_coordinates, view_params):
    """Map each image coordinate to a corresponding direction vector.
    Args:
        pixel (numpy.ndarray): Pixel coordinates of shape (num_pixels, 2)
        view_params (dict): dictionary containing view parameters
    Returns
        (numpy.ndarray): Direction vectors of shape (num_pixels, 3)
    """

    ndc = image_coordinates / np.array([view_params["width"], view_params["height"]])
    direction, extent = fish_eye_polynomial(ndc, view_params)
    view_to_world = view_params["view_to_world"]
    origin = np.matmul(np.array([0, 0, 0, 1]), view_to_world)[:3]
    direction = np.matmul(np.pad(direction, ((0, 0), (0, 1)), constant_values=0), view_to_world)[:, :3]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return origin, direction


def is_valid_projection(points, view_params):
    view_params["aspect_ratio"] = view_params["width"] / view_params["height"]
    view_params["z_near"] = 0.1
    view_params["z_far"] = 1000000
    view_proj_matrix = get_view_proj_mat(view_params)
    homo = np.pad(points, ((0, 0), (0, 1)), constant_values=1.0)
    tf_points = np.dot(homo, view_proj_matrix)
    eps = 1e-5
    if np.any(np.abs(tf_points[..., -1]) <= eps):
        # To avoid division by zero
        return False
    return True


def project_pinhole(points, view_params):
    """
    Project 3D points to 2D camera view using a pinhole camera model.

    Args:
        points (numpy.ndarray): Array of points in world frame of shape (num_points, 3).
        view_params (Dict[str, float]): Dictionary containing viewport parameters.

    Returns:
        (numpy.ndarray): Image-space points of shape (num_points, 3) where the last component is
        the homogeneous z coordinate.
    """
    view_params["aspect_ratio"] = view_params["width"] / view_params["height"]
    view_params["z_near"] = 0.1
    view_params["z_far"] = 1000000

    view_proj_matrix = get_view_proj_mat(view_params)
    homo = np.pad(points, ((0, 0), (0, 1)), constant_values=1.0)
    tf_points = np.dot(homo, view_proj_matrix)

    try:
        tf_points = tf_points / (tf_points[..., -1:])
        tf_points[..., :2] = 0.5 * (tf_points[..., :2] + 1)
    except Exception as error:
        print(f"Exception: Occurred while in project_pinhole: {error} ")
        return None, False

    return tf_points[..., :3], True


def world_to_image(points, view_params):
    """Project world coordinates to image-space.
    Args:
        points (numpy.ndarray): Array of points in world frame of shape (num_points, 3).
        view_params (Dict[str, float]): Dictionary containing viewport parameters.

    Returns:
        (numpy.ndarray): Image-space points of shape (num_points, 3)
    """
    if view_params["projection_type"] == "pinhole" or view_params["projection_type"] is None:
        return project_pinhole(points, view_params)
    elif view_params["projection_type"] == "fisheyePolynomial":
        return project_fish_eye_polynomial(points, view_params)
    else:
        raise ValueError(f"Projection type {view_params['projection_type']} is not currently supported.")


def ftheta_distortion(ftheta, x):
    """F-Theta distortion."""
    return ftheta["poly_a"] + x * (
        ftheta["poly_b"] + x * (ftheta["poly_c"] + x * (ftheta["poly_d"] + x * ftheta["poly_e"]))
    )


def ftheta_distortion_prime(ftheta, x):
    """Derivative to f_theta_distortion."""
    return ftheta["poly_b"] + x * (2 * ftheta["poly_c"] + x * (3 * ftheta["poly_d"] + x * 4 * ftheta["poly_e"]))


def _ftheta_distortion_solver(ftheta, theta):
    # Solve for r in theta = f(r), where f(r) is some general polynomial that is guaranteed to be monotonically
    # increasing up to some maximum r and theta. For theta > maximum theta switch to linear extrapolation.

    def solver(ftheta, theta):
        ratio = ftheta["width"] / 2 / ftheta["edge_fov"]
        guess = theta * ratio

        # 2 loops provides sufficient precision in working range.
        for _i in range(2):
            guessed_theta = ftheta_distortion(ftheta, guess)
            dy = theta - guessed_theta
            dx = ftheta_distortion_prime(ftheta, guess)
            mask = dx != 0
            guess[mask] += dy[mask] / dx[mask]
            guess[~mask] += dy[~mask] * ratio

        return guess

    # For all points guess r using a linear approximation.
    guess = solver(ftheta, theta)

    # Determine which points were actually inside the FOV
    max_theta = math.radians((ftheta["max_fov"] / 2.0))
    inside_fov = theta < max_theta

    # For all points that were outside the FOV replace their solution with a more stable linear extrapolation.
    # These outside of FOV points map beyond the maximum r possible for inside FOV points.
    # These points shouldn't be seen by the camera, but a valid projection is still required.
    max_r = solver(ftheta, np.array([max_theta]))
    min_theta = ftheta["poly_a"]  # this should always be zero in theory, but the user could define poly_a != 0.
    extrapolation_slope = max_r / (max_theta - min_theta)
    guess[~inside_fov] = max_r + extrapolation_slope * (theta[~inside_fov] - max_theta)

    return guess
