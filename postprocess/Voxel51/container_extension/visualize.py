import hashlib
import inspect
import math
import os
import traceback

import cv2
import imageio
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np

# from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf

from .helpers import (
    get_bbox_3d_corners,
    get_sensor_name,
    get_view_params,
    get_view_proj_mat,
    project_fish_eye_polynomial,
    world_to_image,
)

# from pdb import set_trace


# TODO: Fix font location for docker
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
FONT_PATH = os.path.abspath(os.path.join(FILE_DIR, "data/Barlow-Regular.ttf"))

# 3D Cuboids
EGO_INTERSECTION_RADIUS = 5
GREEN_COLOUR = [166, 255, 0]
RED_COLOUR = [255, 0, 0]
PEDESTRIAN_COLOUR = [0, 0, 255]
CUTOFF_DISTANCE = 100
ALLOWED_CLASS = ["person", "automobile", "car", "prop_general", "road_sign", "prop_traffic"]

# Depth
NEAR = 0.01
FAR = 100


def _read_exr(path):
    # Download freeimage dll, will only download once if not present
    # from https://imageio.readthedocs.io/en/stable/format_exr-fi.html#exr-fi
    imageio.plugins.freeimage.download()

    kwargs = {"flags": imageio.plugins.freeimage.IO_FLAGS.EXR_ZIP}
    return imageio.v2.imread(path, format="exr", **kwargs)


def create_video(
    out_dir, gt_subdir, sensor="", framerate=30, fmt="%4d.png", start_num=0, input_dir=None, transparent=False, **kwargs
):
    # Caller should specify a single gt subdir and sensor
    # can be called in loop to get all

    # Transparent videos case
    if transparent and kwargs.get("view") is not None and len(input_dir) > 0:
        output_file_path = f"{out_dir}/{sensor}.mov"
        view = kwargs.get("view")
        output_file_path = f"{out_dir}/{view}_{sensor}.mov"
        # Create transparent argb video. Note: red background may need to use yuv420p instead.
        command = (
            f"ffmpeg -framerate {framerate} -start_number {start_num} -i {input_dir} "
            f" -vcodec qtrle -pix_fmt argb "
            f"{output_file_path}"
        )

    # Non-transparent videos case - RGB images, post-processed images, etc
    else:
        output_file_path = f"{out_dir}/{sensor}.mp4"

        # RGB images case
        if input_dir:
            images_pattern = f"{input_dir}/{fmt}"
            command = (
                f"ffmpeg -f image2 -r {framerate} -start_number {start_num} -y "
                f"-i {images_pattern} -vcodec libx264 -crf 18  -pix_fmt yuv420p "
                f"{output_file_path}"
            )
        else:
            # Lidar view case - top down/first person view
            if kwargs.get("view") is not None:
                view = kwargs.get("view")
                output_file_path = f"{out_dir}/{sensor}_{view}.mp4"
                command = (
                    f"ffmpeg -f image2 -r {framerate} -start_number {start_num} -y "
                    f"-i {out_dir}/{gt_subdir}/{sensor}/{view}/{fmt} -vcodec libx264 -crf 18  -pix_fmt yuv420p "
                    f"{output_file_path}"
                )
            # Post-processed images case
            else:
                command = (
                    f"ffmpeg -f image2 -r {framerate} -start_number {start_num} -y "
                    f"-i {out_dir}/{gt_subdir}/{sensor}/{fmt} -vcodec libx264 -crf 18  -pix_fmt yuv420p "
                    f"{output_file_path}"
                )
    try:
        exitcode = os.system(command)
    except Exception as e:
        print(f"Video could not be created! Maybe dependencies are missing? Failed with exception: \n {e}")
        traceback.print_exc()
    if exitcode != 0:
        print(f"Video creation FAILED. Exited with code: {exitcode}")
        print("NOTE: If using docker, try video creation on host. Command for video creation is:")
        print(command)

    return output_file_path


# TODO: Integrate this fully with post-processing
def create_video_cv2(src_dir, gt_subdir, sensor="", framerate=30, **kwargs):
    images = [img for img in os.listdir(f"{src_dir}/{gt_subdir}/{sensor}") if img.endswith(".png")]
    images = sorted(images, key=lambda x: int(os.path.splitext(x)[0]))

    # TODO: Dynamically adjust image size.
    video = cv2.VideoWriter(
        f"{src_dir}/{gt_subdir}_{sensor}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), framerate, (1920, 1080)
    )

    for image in images:
        image = os.path.join(src_dir, gt_subdir, sensor, image)
        frame = cv2.imread(image)

        frame = cv2.resize(frame, (1920, 1080))
        video.write(frame)

    video.release()


def data_to_colour(data):
    if isinstance(data, str):
        data = bytes(data, "utf-8")
    else:
        data = bytes(data)
    m = hashlib.sha256()
    m.update(data)
    key = int(m.hexdigest()[:8], 16)
    r = ((((key >> 0) & 0xFF) + 1) * 33) % 255
    g = ((((key >> 8) & 0xFF) + 1) * 33) % 255
    b = ((((key >> 16) & 0xFF) + 1) * 33) % 255

    # illumination normalization to 128
    inv_norm_i = 128 * (3.0 / (r + g + b))

    return (r * inv_norm_i, g * inv_norm_i, b * inv_norm_i)

    # hsl = (int(m.hexdigest(), 16) % 359 / 360.), 0.5, 0.5    # 359 is a prime number
    # return colorsys.hls_to_rgb(*hsl)


def _draw_lollipop(draw, bbox_points, distance, velocity, colour, w, h):
    def _draw_circle(draw, centre, radius, fill=None, outline=None, width=3, start=None, end=None):
        circle_p0 = centre[0] - radius, centre[1] - radius
        circle_p1 = centre[0] + radius, centre[1] + radius
        if start is None or end is None:
            draw.ellipse([circle_p0, circle_p1], fill=fill, outline=outline, width=width)
        else:
            draw.arc([circle_p0, circle_p1], fill=outline, start=start, end=end, width=width)

    alpha_sf = max(0, 1 - max(0, distance - CUTOFF_DISTANCE) / CUTOFF_DISTANCE)  # scale alpha by distance
    alpha_sf *= min(1.0, velocity)  # scale alpha by velocity
    sf = w / 3848
    bbox_size = np.linalg.norm(np.min(bbox_points, axis=0) - np.max(bbox_points, axis=0))
    sf *= min(1.0, max(0.5, sf * bbox_size / 2000))  # Scale lollipop by perceptual car size
    lollipop_stem_length = 100 * sf
    circle0 = 90 * sf
    circle1 = 70 * sf
    circle2 = 40 * sf
    arc = 63 * sf
    # flag_width_scale = 8
    # flag_height = 120

    # Shade bbox
    face_idx_list = [[0, 1, 3, 2], [4, 5, 7, 6], [2, 3, 7, 6], [0, 1, 5, 4], [0, 2, 6, 4], [1, 3, 7, 5]]
    face_idx_list.pop(2)  # Remove bottom face

    for face_idxs in face_idx_list:
        draw.polygon(
            [tuple(xy) for xy in bbox_points[face_idxs]], fill=tuple([*colour, int(40 * alpha_sf)])  # noqa: C409
        )

    p0 = tuple(bbox_points[0])  # top front
    p1 = tuple(bbox_points[1])  # top front
    p2 = tuple(bbox_points[4])  # top back
    p3 = tuple(bbox_points[5])  # top back

    stem_base = (p0[0] + p1[0] + p2[0] + p3[0]) / 4, (p0[1] + p1[1] + p2[1] + p3[1]) / 4
    stem_tip = stem_base[0], max(0, stem_base[1] - lollipop_stem_length)

    # Draw flag
    # stem_base_b = (p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2
    # direction = 1 if stem_base[0] < stem_base_b[0] else -1
    # flag_p0 = stem_base[0], stem_base[1] - lollipop_stem_length / 2 + flag_height / 2
    # flag_p1 = stem_base[0], stem_base[1] - lollipop_stem_length / 2 - flag_height / 2
    # flag_p2 = stem_base[0] + direction * flag_width_scale * velocity, stem_base[1] \
    #  - lollipop_stem_length / 2 + flag_height / 2
    # flag_p3 = stem_base[0] + direction * flag_width_scale * velocity, stem_base[1] \
    #  - lollipop_stem_length / 2 - flag_height / 2
    # draw.polygon([flag_p0, flag_p1, flag_p3, flag_p2], fill=(*colour, 200))

    # Draw lollipop background
    lollipop_centre = stem_tip[0], stem_tip[1] - circle1
    _draw_circle(draw, lollipop_centre, radius=circle0, fill=(50, 50, 50, int(80 * alpha_sf)))

    # Draw lollipop stem
    draw.line([stem_base, stem_tip], width=int(3 * sf), fill=tuple([*colour, int(255 * alpha_sf)]))  # noqa: C409

    # Draw lollipop
    # Draw outline + gauge background
    _draw_circle(
        draw,
        lollipop_centre,
        radius=circle1,
        fill=(*[int(c * 0.2) for c in colour], int(120 * alpha_sf)),
        outline=(*colour, int(200 * alpha_sf)),
        width=int(3 * sf),
    )
    # Draw text background
    _draw_circle(
        draw,
        lollipop_centre,
        radius=circle2,
        fill=(*[int(c * 0.7) for c in colour], int(120 * alpha_sf)),
        width=int(5 * sf),
    )
    # Draw gauge
    _draw_circle(
        draw,
        lollipop_centre,
        radius=arc,
        outline=(*colour, int(200 * alpha_sf)),
        width=int(18 * sf),
        start=-90,
        end=-90 + velocity / 100 * 360,
    )
    # Draw top line
    draw.line(
        [(stem_tip[0], stem_tip[1] - 175 * sf), (stem_tip[0], stem_tip[1] - 116 * sf)],
        fill=(*colour, int(200 * alpha_sf)),
        width=int(3 * sf),
    )

    draw_scale = 35 * sf
    font = ImageFont.truetype(FONT_PATH, size=int(round(draw_scale)))
    text = str(round(velocity, 1))
    # text_size = font.getsize(text)
    # Use getbbox to calculate the size of the text instead of the deprecated font.getsize
    text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # text_size is now a tuple (width, height)
    text_size = (text_width, text_height)

    draw.text(
        [lollipop_centre[0] - text_size[0] / 2, lollipop_centre[1] - text_size[1] / 1.5],
        text,
        font=font,
        # anchor="mm",
        stroke_fill=(255, 255, 255, int(255 * alpha_sf)),
        stroke_width=1,
    )


def _ray_sphere_intersection(origin, direction, radius, sphere_origin=None):
    if sphere_origin is not None:
        origin = origin - sphere_origin

    # Check if car direction of travel hits ego sphere
    a = 1
    b = 2 * np.dot(direction, origin)
    c = np.dot(origin, origin) - radius**2

    return np.any(np.roots([a, b, c]) > 0)


def _draw_arrow(draw, points_local, distance, velocity, sensor_params, colour, w, h):
    distance_from_box = 0.1
    pointiness = 0.05
    thickness = 0.05

    alpha_sf = max(0, 1 - max(0, distance - CUTOFF_DISTANCE) / CUTOFF_DISTANCE)  # scale alpha by distance
    alpha_sf *= min(1.0, velocity)  # scale alpha by velocity

    bfl, bfr = points_local[2], points_local[3]  # bottom front left, bottom front right
    bbl, bbr = points_local[6], points_local[7]  # bottom back left, bottom back right
    bfm = (bfl + bfr) / 2  # bottom front middle, bottom front middle
    bbm = (bbl + bbr) / 2  # bottom front middle, bottom front middle

    dir_l = bfl - bbl
    dir_l /= np.linalg.norm(dir_l)
    dir_r = bfr - bbr
    dir_r /= np.linalg.norm(dir_r)
    dir_m = bfm - bbm
    dir_m /= np.linalg.norm(dir_m)

    sign = -1 if bfl[0] > bbl[0] else 1

    if sign < 0:
        bl, bm, br = bbl, bbm, bbr
    else:
        bl, bm, br = bfl, bfm, bfr

    distance_to_front = np.linalg.norm(bfm)

    arrow_points = np.array(
        [
            bl + sign * dir_l * distance_to_front * distance_from_box,
            bm + sign * dir_m * distance_to_front * (distance_from_box + sign * pointiness),
            br + sign * dir_r * distance_to_front * distance_from_box,
            br + sign * dir_r * distance_to_front * (distance_from_box + thickness),
            bm + sign * dir_m * distance_to_front * (distance_from_box + sign * pointiness + thickness),
            bl + sign * dir_l * distance_to_front * (distance_from_box + thickness),
        ]
    )

    projected, status = world_to_image(arrow_points[..., :3].reshape(-1, 3), sensor_params)
    if status:
        projected = projected[..., :3].reshape(-1, 3)
        projected_points = projected[..., :2] * np.array([[w, h]])
        draw.polygon([tuple(ap) for ap in projected_points.tolist()], fill=(*colour, int(200 * alpha_sf)))


def _draw_wireframe(
    img,
    bboxes_3d_corners,
    semantic_labels,
    distances,
    velocities,
    furthest=None,
    line_colour=None,
):
    face_idx_list = [[0, 1, 3, 2], [4, 5, 7, 6], [2, 3, 7, 6], [0, 1, 5, 4], [0, 2, 6, 4], [1, 3, 7, 5]]
    face_idx_list.pop(2)  # Remove bottom face
    w, h = img.size

    overlay = Image.fromarray(np.zeros((h, w, 4), dtype=np.uint8))
    sf = overlay.width / 3848
    draw = ImageDraw.Draw(overlay)
    # colour = [118, 185, 0]
    if line_colour is None:
        line_colour = [200, 200, 200]
    for _i, (p, d, v, s) in enumerate(zip(bboxes_3d_corners, distances, velocities, semantic_labels)):
        alpha_sf = max(0, 1 - max(0, d - CUTOFF_DISTANCE) / CUTOFF_DISTANCE)  # scale alpha by distance
        v = round(v, 1)
        for _j, face_idxs in enumerate(face_idx_list):
            if "person" in s:
                draw.polygon([tuple(xy) for xy in p[face_idxs]], fill=tuple([*PEDESTRIAN_COLOUR, 40]))  # noqa: C409
            if any(label in s for label in ALLOWED_CLASS):
                # DRIVE-11792: removed until we can expose param to CLI
                # We should not filter by default bc it is confusing to users
                # if s in ["prop_general", "road_sign"] and d > distance_filter:
                #    continue
                draw.line(
                    [tuple(xy) for xy in p[face_idxs]],
                    width=int(3 * sf),
                    fill=tuple([*line_colour, int(200 * alpha_sf)]),  # noqa: C409
                )
                draw.line(
                    [tuple(p[face_idxs[0]]), tuple(p[face_idxs[-1]])],
                    width=int(3 * sf),
                    fill=tuple([*line_colour, int(200 * alpha_sf)]),  # noqa: C409
                )
    # overlay = overlay.resize((w, h), Image.ANTIALIAS)
    return Image.alpha_composite(img, overlay)


# Function to generate colors for each point based on squared distance
def generate_colors_from_distance(points):
    """
    Generate an array of colors for each point in the point cloud based on the distance from the origin.

    Args:
    - points (numpy.ndarray): The point cloud as an Nx3 numpy array.

    Returns:
    - numpy.ndarray: An Nx3 array of RGB colors for each point.
    """
    # Calculate distances from the origin
    distances = np.linalg.norm(points, axis=1)

    # Apply a non-linear transformation to make color differences more noticeable
    # Here, we use a simple square root, but you can experiment with other functions
    # distances = np.sqrt(distances)

    # Normalize distances to [0, 1] for color mapping
    normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())

    # Use a colormap (e.g., viridis) and map normalized distances to colors
    # colormap = cm.get_cmap("viridis")
    # colormap = cm.get_cmap('magma_r')
    # colormap = cm.get_cmap('plasma')
    # colormap = cm.get_cmap('hot')

    # Define a custom colormap from red to green
    #               red            orange             yellow           green-yellow       green
    colors = [(0, (1, 0, 0)), (0.15, (1, 0.64, 0)), (0.25, (1, 1.0, 0)), (0.35, (0.10, 1, 0)), (1, (0, 1, 0))]
    colormap = mpl_colors.LinearSegmentedColormap.from_list("custom_hot_to_green", colors, N=200)

    colors = colormap(normalized_distances)[:, :4]  # Get RGB components, ignore alpha
    colors *= 255

    return colors


def _extract_lidar(data):
    lidar_xyz = data["lidar"][:, 0:3]

    # Green point cloud.
    # colours = np.zeros((len(lidar_xyz), 4), dtype=np.uint8)
    # lidar_intensity = data["lidar"]["intensity"]
    # colours[:, 1] = 255 * lidar_intensity * 10
    # colours[:, 3] = 255

    colours = generate_colors_from_distance(lidar_xyz)

    return lidar_xyz, colours


def clip_to_bounds(projected_pc, width, height):
    bound_x = np.logical_and(projected_pc[:, 0] >= 0, projected_pc[:, 0] < width)
    bound_y = np.logical_and(projected_pc[:, 1] >= 0, projected_pc[:, 1] < height)
    bound_z = np.logical_and(projected_pc[:, 2] >= 0, projected_pc[:, 2] < 1.0)
    bound_mask = np.logical_and(bound_x, bound_y, bound_z)
    projected_pc = projected_pc[bound_mask]
    return bound_mask, projected_pc


def _draw_sensor_locations(view_params, sensors_in_ego_coords):
    w, h = view_params["width"], view_params["height"]

    # Pre-pend the sensor positions to the points
    sensors_pos = None
    for key in sensors_in_ego_coords:
        value_arr = [np.array(sensors_in_ego_coords[key], dtype=float)]
        if sensors_pos is None:
            sensors_pos = value_arr
        else:
            sensors_pos = np.concatenate((sensors_pos, value_arr))

    # Note: world_to_image here is not the right name.  Read sensor_to_image.
    projected_sensors_pos, status = world_to_image(sensors_pos, view_params)
    if status:
        projected_sensors_pos = (projected_sensors_pos * np.array([[w, h, 1]])).astype(int)

        sensor_pos_overlay = np.zeros((h, w, 4), dtype=np.uint8)
        num_sensors = len(sensors_pos)

        # Hot pink
        color = np.array([255.0, 105.0, 180.0, 255.0], dtype=np.float)
        colors = np.tile(color, (num_sensors, 1))

        # Make the projected points bigger
        projected_sensors_pos, colors = _expand_points(projected_sensors_pos, colors, w, h)

        sensor_pos_overlay[projected_sensors_pos[:, 1], projected_sensors_pos[:, 0]] = colors
        return Image.fromarray(sensor_pos_overlay), True

    return None, False


def _expand_points(projected_pc, colours, w, h):
    # Make the projected points bigger
    offsets = np.array([[1, 1, 0], [1, 0, 0], [1, -1, 0], [0, 1, 0], [0, -1, 0], [-1, 1, 0], [-1, 0, 0], [-1, -1, 0]])
    orig_projected_pc = projected_pc.copy()
    color_indexes = np.arange(len(projected_pc))
    orig_color_indexes = color_indexes.copy()

    for offset in offsets:
        new_projected_pc = orig_projected_pc + offset
        projected_pc = np.concatenate((projected_pc, new_projected_pc))
        color_indexes = np.concatenate((color_indexes, orig_color_indexes))

    bound_mask, projected_pc = clip_to_bounds(projected_pc, w, h)
    color_indexes = color_indexes[bound_mask]
    return projected_pc, colours[color_indexes]


def get_valid_pinhole_projections(points, view_params):
    """
    Project 3D points to 2D camera view using a pinhole camera model.

    Args:
        points (numpy.ndarray): Array of points in world frame of shape (num_points, 3).
        view_params (Dict[str, float]): Dictionary containing viewport parameters.

    Returns:
        (numpy.ndarray): Image-space points of shape (num_points, 3)
    """
    view_params["aspect_ratio"] = view_params["width"] / view_params["height"]
    view_params["z_near"] = 0.1
    view_params["z_far"] = 1000000

    view_proj_matrix = get_view_proj_mat(view_params)
    homo = np.pad(points, ((0, 0), (0, 1)), constant_values=1.0)
    tf_points = np.dot(homo, view_proj_matrix)

    # Set the points that are incorrectly projected to zero.  Note that we then
    # need to avoid using these points to divide by w.
    tf_points[tf_points[:, -1] >= 0] *= 0

    # Create a mask to filter out rows with 0 in the last column
    valid_mask = tf_points[:, -1] != 0

    # Filter out those rows
    valid_tf_points = tf_points[valid_mask]

    # Now perform the division safely on valid_tf_points
    valid_tf_points = valid_tf_points / valid_tf_points[:, -1, None]

    valid_tf_points[..., :2] = 0.5 * (valid_tf_points[..., :2] + 1)

    return valid_tf_points[..., :3], valid_mask


def _draw_pointcloud(points, view_params, colours, is_radar=False):
    w, h = view_params["width"], view_params["height"]

    # Note: world_to_image here is not the right name.  Read sensor_to_image.
    projected_pc, valid_mask = get_valid_pinhole_projections(points, view_params)
    projected_pc = (projected_pc * np.array([[w, h, 1]])).astype(int)

    pc_overlay = np.zeros((h, w, 4), dtype=np.uint8)
    bound_mask, projected_pc = clip_to_bounds(projected_pc, w, h)

    if colours is None:
        # colours = np.full((len(projected_pc), 4), [0, 0, 255, 255], dtype=np.uint8)
        colours = get_colors_for_points(points)

    colours = colours[valid_mask]

    if not isinstance(colours, tuple):
        colours = colours[bound_mask]

    projected_pc, colours = _expand_points(projected_pc, colours, w, h)

    pc_overlay[projected_pc[:, 1], projected_pc[:, 0]] = colours
    return Image.fromarray(pc_overlay)


def colorize_segmentation(segmentation_image, instance_mapping, semantic=False):
    # segmentation_ids = np.unique(segmentation_image)

    keys = sorted([int(k) for k in instance_mapping])

    lut = np.array([[0] + keys, [0] + list(range(1, len(instance_mapping) + 1))])
    new_segmentation_image = lut[1, np.searchsorted(lut[0, :], segmentation_image)]
    colours = np.array([[0, 0, 0, 0]] + [[*data_to_colour(str(instance_mapping[str(k)])), 255] for k in keys])
    segmentation_image = (colours[new_segmentation_image]).astype(int)
    return (segmentation_image).astype(np.uint8)


def show_depth(backend, data, out_dir=""):
    # Write depth in greyscale
    fpath = os.path.join(out_dir, "depth", get_sensor_name(data), f"{data['frame_id']:04d}.png")
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    depth_data = data["depth_linear"]

    depth_data = np.clip(depth_data, NEAR, FAR)
    # depth_data = depth_data / far
    depth_data = (np.log(depth_data) - np.log(NEAR)) / (np.log(FAR) - np.log(NEAR))
    # depth_data = depth_data / far
    depth_data = 1.0 - depth_data

    depth_data_uint8 = (depth_data * 255).astype(np.uint8)
    depth_data_colour = cv2.applyColorMap(depth_data_uint8, cv2.COLORMAP_JET)

    img = Image.fromarray(depth_data_colour)

    # rgb = Image.fromarray(data["rgb"])
    # rgb.convert("L")
    # rgb.convert("RGB")
    # img = Image.blend(rgb, img, 0.98)

    # buf = io.BytesIO()
    # img.save(buf, format="png", compress_level=1)
    # backend.write_blob(fpath, buf.getvalue())

    # OpenCV encoding is +30% faster
    arr = np.asarray(img)[..., [2, 1, 0]]
    success, buf = cv2.imencode(".png", arr)
    buf = buf.tobytes()
    backend.write_blob(fpath, buf)


def transform_same(bbox_transform, ego_transform):
    # compare element by element
    eps = 0.001
    transform_mat_dim = 4
    k = 0

    for i in range(transform_mat_dim):
        for j in range(transform_mat_dim):
            if abs(bbox_transform[i][j] - ego_transform[k] > eps):
                return False
            k = k + 1
    return True


def show_bounding_boxes_3d(backend, data, out_dir=""):
    # Visualize 3D BBoxes
    img = Image.fromarray(data["rgb"]).convert("RGBA")
    h, w = data["rgb"].shape[:2]

    save_bounding_boxes_3d(backend, data, out_dir, img, w, h)


def save_bounding_boxes_3d(backend, data, out_dir, image_from_array, w, h, outdir_as_filepath=False):
    # Get a draw scale based on resolution
    # draw_scale = w / 1000.0
    # thickness_std = max(1, int(1 * draw_scale))
    img = image_from_array

    # overlay = Image.new("RGBA", (w * 4, h * 4)) # make overlay 4 times larger to use supersampling for antialiasing
    overlay = Image.fromarray(np.zeros((h * 4, w * 4, 4), dtype=np.uint8))

    draw = ImageDraw.Draw(overlay)

    points_local = []
    velocities = []
    semantics = []
    sensor_tf_inv = np.array(data["camera_params"]["cameraViewTransform"]).reshape((4, 4))
    # sensor_tf = np.linalg.inv(sensor_tf_inv)  # TODO: Document and explain this

    bbox3d_corners = get_bbox_3d_corners(data["bounding_box_3d"])
    bbox3d_mapping = data["bounding_box_3d_mapping"]

    # Extract ego transform from data
    # ego_transform = data["transforms"]["ego_transform"]["egoTransform"]
    for bbox3d_idx, (bbox3d, prim_path) in enumerate(zip(data["bounding_box_3d"], data["bounding_box_3d_paths"])):
        semantic_id = str(bbox3d["semanticId"])
        semantic_class = bbox3d_mapping[semantic_id].get("class")
        if semantic_class is None:
            print(f"Skipping 3D bbox for prim '{prim_path}', as it has no semantic class!")
            continue
        elif "Ego" in prim_path:
            print(f"Found suspected ego prim in 3D bbox data, skipping 3D bbox for prim: {prim_path}")
            continue

        corners_h = np.pad(bbox3d_corners[bbox3d_idx], ((0, 0), (0, 1)), constant_values=1)
        corners_sensor_frame = np.einsum("jk,kl->jl", corners_h, sensor_tf_inv)

        # CENTRE
        centre = corners_sensor_frame.mean(axis=0)[:3]
        x = centre[0]
        y = centre[1]
        z = centre[2]

        # DIMENSIONS
        scale = np.linalg.norm(bbox3d["transform"][:3], axis=1)
        d_x = (bbox3d["x_max"] - bbox3d["x_min"]) * scale[0] * 0.5
        d_y = (bbox3d["y_max"] - bbox3d["y_min"]) * scale[1] * 0.5
        d_z = (bbox3d["z_max"] - bbox3d["z_min"]) * scale[2] * 0.5

        # ORIENTATION
        # obj2sensor_tf = np.matmul(bbox3d["transform"], sensor_tf_inv)
        # obj2sensor_tf = Gf.Matrix3d(obj2sensor_tf[:3, :3]).GetOrthonormalized()
        # obj_rot_sensor = obj2sensor_tf.ExtractRotation()
        # roll_pitch_yaw = obj_rot_sensor.Decompose(Gf.Vec3d([1, 0, 0]), Gf.Vec3d([0, 1, 0]), Gf.Vec3d([0, 0, 1]))
        # o_x = roll_pitch_yaw[0]
        # o_y = roll_pitch_yaw[1]
        # o_z = roll_pitch_yaw[2]

        # Project to camera
        corners = np.array(
            [
                [d_x, -d_y, d_z],
                [d_x, d_y, d_z],
                [d_x, -d_y, -d_z],
                [d_x, d_y, -d_z],
                [-d_x, -d_y, d_z],
                [-d_x, d_y, d_z],
                [-d_x, -d_y, -d_z],
                [-d_x, d_y, -d_z],
            ]
        )

        # We can set the rotation matrix from the orientation angles or the transforms:
        # rot = Gf.Rotation([0, 0, 1], o_z) * Gf.Rotation([1, 0, 0], o_x) * Gf.Rotation([0, 1, 0], o_y)
        # rot_mat = Gf.Matrix3d(rot)
        # obj_tf = np.array(Gf.Matrix4d().SetRotate(rot_mat).SetTranslateOnly(Gf.Vec3d([x, y, z])))
        # --
        # Using the rotation matrix from the transforms to avoid cases where the angles do not recover the original tf
        obj_tf = np.matmul(bbox3d["transform"], sensor_tf_inv)
        obj_tf[3, :3] = [x, y, z]

        corners_h = np.pad(corners, ((0, 0), (0, 1)), constant_values=1)
        # points_local = np.einsum("jk,kl->jl", corners_h, obj_tf)
        points_local.append(np.einsum("jk,kl->jl", corners_h, obj_tf))

        velocity = [0.0, 0.0, 0.0]
        if "dynamics" in data and prim_path in data["dynamics"]["vehicle_dynamics"]["primPaths"]:
            idx = data["dynamics"]["vehicle_dynamics"]["primPaths"].index(prim_path)
            velocity = data["dynamics"]["vehicle_dynamics"]["primVelocities"][idx]

        velocities.append(velocity)
        semantics.append(semantic_class)

    if len(points_local) > 0:
        velocities = np.array(velocities)
        points_local = np.array(points_local)
        data["render_products"]["resolution"] = [w, h]
        view_params = get_view_params(data)
        view_params["world_to_view"] = np.eye(4)
        # Returns success status
        points_image, success = project_fish_eye_polynomial(points_local[..., :3].reshape(-1, 3), view_params)
        if not success:
            print("Fisheye projection failed!")
            return
        projected = points_image[..., :3].reshape(-1, 8, 3)
        projected_corners = projected[..., :2].reshape(-1, 8, 2)
        distance_3d_corners = np.linalg.norm(points_local.reshape(-1, 4)[:, :3], axis=1).reshape(-1, 8)
        sort_keys = np.argsort(np.max(distance_3d_corners, axis=1))[::-1]
        projected_corners = projected_corners[sort_keys]
        points_local = points_local[sort_keys]
        furthest = np.argsort(distance_3d_corners[sort_keys], axis=1)[:, -1]
        velocities = velocities[sort_keys]
        distances = np.max(distance_3d_corners, axis=1)[sort_keys]
        semantics = np.array(semantics)[sort_keys]

        velocity_magnitudes = np.linalg.norm(velocities[..., :3], axis=1) * 3.6  # m/s -> km/h

        overlay_width, overlay_height = overlay.size
        projected_corners *= np.array([[overlay_width, overlay_height]])

        for i, box in enumerate(projected_corners):
            velocity_magnitude = velocity_magnitudes[i]
            if round(velocity_magnitude, 1) > 0:
                vehicle_front = (points_local[i][2] + points_local[i][3]) / 2
                vehicle_back = (points_local[i][6] + points_local[i][7]) / 2
                vehicle_dir = vehicle_front - vehicle_back
                vehicle_dir /= np.linalg.norm(vehicle_dir)
                is_intersecting = _ray_sphere_intersection(vehicle_front[:3], vehicle_dir[:3], EGO_INTERSECTION_RADIUS)
                colour = RED_COLOUR if is_intersecting else GREEN_COLOUR
                _draw_lollipop(draw, box, distances[i], velocity_magnitude, colour, overlay_width, overlay_height)
                _draw_arrow(
                    draw,
                    points_local[i],
                    distances[i],
                    velocity_magnitude,
                    view_params,
                    colour,
                    overlay_width,
                    overlay_height,
                )
        overlay = _draw_wireframe(overlay, projected_corners, semantics, distances, velocity_magnitudes, furthest)

        overlay = overlay.resize([w, h], Image.LANCZOS)
        img = Image.alpha_composite(img, overlay)

    if outdir_as_filepath:
        fpath = out_dir
        img_format = ".jpeg"
    else:
        fpath = os.path.join(out_dir, "3d_cuboids", get_sensor_name(data), "{:04d}{}".format(data["frame_id"], ".png"))
        img_format = ".png"

    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    # buf = io.BytesIO()
    # img.save(buf, format="png", compress_level=1)
    # backend.write_blob(fpath, buf.getvalue())

    # OpenCV encoding is +30% faster
    arr = np.asarray(img)[..., [2, 1, 0, 3]]
    success, buf = cv2.imencode(img_format, arr)
    buf = buf.tobytes()
    backend.write_blob(fpath, buf)


def create_camera_view_params(view, width, height):
    view_params = {
        "projection_type": "pinhole",
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "clipping_range": [0.001, 1000000.0],
        "horizontal_aperture": 20.955,
        "focal_length": 15.2908,
        # 'projection_type': 'fisheyePolynomial',
        # 'ftheta': {'width': w, 'height': h, 'cx': 961.6477661132812, 'cy': 599.9234619140625, \
        # 'poly_a': 0.0, 'poly_b': 0.002320797648280859, 'poly_c': -5.9293267185012155e-08, \
        # 'poly_d': -1.0979823716894543e-09, 'poly_e': 6.905446266886051e-13, 'max_fov': 200.0, \
        # 'edge_fov': 1.801367655660413, 'c_ndc': np.array([-0.00328111,  0.00417177])},
        # 'width': w,
        # 'height': h,
        # 'aspect_ratio': w / h,
    }

    if view == "first_person":
        view_params["world_to_view"] = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, -5, 0, 1]])
    elif view == "top_down" or view == "birdseye":
        view_params["world_to_view"] = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, -80, 1]])
    elif view == "third_person":
        view_params["world_to_view"] = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, -5, -10, 1]])
    else:
        raise ValueError(f"View argument `{view}` is invalid. Choose from [first_person, top_down, third_person]")
        return {}

    return view_params


def get_sensor_pos_overlay(data, view_params):
    sensors_in_ego_coords = data["translations"]
    return _draw_sensor_locations(view_params, sensors_in_ego_coords)


def get_point_cloud_overlay(data, view_params, width, height):
    if "lidar" in data:
        points, colours = _extract_lidar(data)
        return _draw_pointcloud(points, view_params, colours)
    elif "radar" in data:
        points = data["radar"]
        # colours = np.full((points.shape[0], 4), [0, 255, 0, 255], dtype=np.uint8)
        return _draw_pointcloud(points, view_params, None, is_radar=True)
    elif "vdb_points" in data:
        return _draw_pointcloud(data["vdb_points"], view_params, data["vdb_colors"], is_radar=True)

    return np.zeros((height, width, 4), dtype=np.uint8)


def get_colors_for_points(points):
    def calculate_length(point):
        return np.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)

    # Calculate the lengths of the points
    lengths = np.apply_along_axis(calculate_length, 1, points)

    # Define the color map
    cmap = plt.cm.get_cmap("viridis")  # You can choose other colormaps as well

    # Normalize the lengths to the range [0, 1] for colormap mapping and sort in descending order.
    normalized_lengths = (lengths - np.min(lengths)) / (np.max(lengths) - np.min(lengths))

    # Invert the normalized lengths to prioritize brighter colors for smaller lengths
    inverted_normalized_lengths = 1 - normalized_lengths

    # Map the inverted normalized lengths to colors using the colormap
    return cmap(inverted_normalized_lengths) * 255


# Show the bounding boxes for a given frame
def save_bbox_and_pc_to_image(backend, bboxes, data, view="first_person", out_dir=""):
    # Visualize 3D BBoxes
    h, w = data["rgb_height"], data["rgb_width"]
    black_colors = np.zeros((h, w, 4), dtype=np.uint8) * 255
    black_colors[..., 3] = 255
    img = Image.fromarray(black_colors).convert("RGBA")

    # Get a draw scale based on resolution
    # draw_scale = w / 1000.0
    # thickness_std = max(1, int(1 * draw_scale))

    # Get Camera Orientation
    bbox_overlay = Image.fromarray(np.zeros((h * 4, w * 4, 4), dtype=np.uint8))

    draw = ImageDraw.Draw(bbox_overlay)

    # View params and frame id are needed for both bboxes and pc.
    view_params = create_camera_view_params(view, w, h)
    frame_id = data["frame_id"]

    # print(f"Thread id: {threading.get_ident()} ")
    try:
        if bboxes:
            centroids = []
            points_local = []
            velocities = []
            semantics = []

            # print(f"Invoked save_bbox_and_pc_to_image for frame: {frame_id}")
            for bbox in bboxes:
                cuboid_3d = bbox["shape3d"]["cuboid3d"]
                center = cuboid_3d["center"]
                orientation = cuboid_3d["orientation"]
                dimensions = cuboid_3d["dimensions"]
                # color = (255, 0, 0)

                # Project to camera

                # Reconstitute the bbox from the data in the .json file.
                x = center[0]
                y = center[1]
                z = center[2]
                d_x = dimensions[0] * 0.5
                d_y = dimensions[1] * 0.5
                d_z = dimensions[2] * 0.5
                o_x = math.degrees(orientation[0])
                o_y = math.degrees(orientation[1])
                o_z = math.degrees(orientation[2])

                centroids.append([x, y, z])
                corners = np.array(
                    [
                        [d_x, -d_y, d_z],
                        [d_x, d_y, d_z],
                        [d_x, -d_y, -d_z],
                        [d_x, d_y, -d_z],
                        [-d_x, -d_y, d_z],
                        [-d_x, d_y, d_z],
                        [-d_x, -d_y, -d_z],
                        [-d_x, d_y, -d_z],
                    ]
                )

                # The rotations should be wrt the ego.
                rot = Gf.Rotation([1, 0, 0], o_x) * Gf.Rotation([0, 1, 0], o_y) * Gf.Rotation([0, 0, 1], o_z)
                rot_mat = Gf.Matrix3d(rot)
                obj_tf = np.array(Gf.Matrix4d().SetRotate(rot_mat).SetTranslateOnly(Gf.Vec3d([x, y, z])))

                corners_h = np.pad(corners, ((0, 0), (0, 1)), constant_values=1)

                # Note: These point_in_sensor_coord should be in sensor coordinates.
                point_in_sensor_coord = np.einsum("jk,kl->jl", corners_h, obj_tf)

                # Project to sensor image to see if they are valid bboxes.
                # Attention: According to JF: world_to_image is actually sensor_to_image
                status = False
                try:
                    projected_point_in_sensor_coord, status = world_to_image(
                        point_in_sensor_coord[..., :3].reshape(-1, 3), view_params
                    )
                    if not status:
                        continue

                    projected_point_in_sensor_coord = projected_point_in_sensor_coord[..., :3].reshape(-1, 8, 3)
                except Exception as error:
                    print(f"Exception: Occurred in world_to_image: {error}")
                    traceback.print_exc()
                    continue

                # print(f"In save_bbox_and_pc_to_image: Adding {point_in_sensor_coord}")
                points_local.append(point_in_sensor_coord)
                # Reading HACK: pmaciel
                # GT does not have semanticId to get mapping
                # Changes needed on writer side to export this
                # Until then, we classify all as 'automobile' in bounding boxes
                # We don't display class on bounding boxes, so this is OK
                semantics.append("automobile")

                velocity = bbox["shape3d"]["velocity"]
                # print("velocity: ", velocity)
                velocities.append(tuple(velocity))

            # Note: centroids don't appear to be used here
            centroids = np.array(centroids)
            velocities = np.array(velocities)
            points_local = np.array(points_local)

            # view_params = create_camera_view_params(view, w, h)
            projected, status = world_to_image(points_local[..., :3].reshape(-1, 3), view_params)
            if status and len(projected[..., :3]) % 8 == 0:
                try:
                    projected = projected[..., :3].reshape(-1, 8, 3)
                    projected_corners = projected[..., :2].reshape(-1, 8, 2)
                except Exception as error:
                    print(f"Exception: Occurred in world_to_image: {error}")
                    traceback.print_exc()

                distance_3d_corners = np.linalg.norm(points_local.reshape(-1, 4)[:, :3], axis=1).reshape(
                    -1, 8
                )  # Distance to each point
                sort_keys = np.argsort(np.max(distance_3d_corners, axis=1))[::-1]
                projected_corners = projected_corners[sort_keys]
                points_local = points_local[sort_keys]
                furthest = np.argsort(distance_3d_corners[sort_keys], axis=1)[:, -1]

                velocities = velocities[sort_keys]
                distances = np.max(distance_3d_corners, axis=1)[sort_keys]
                semantics = np.array(semantics)[sort_keys]

                velocity_magnitudes = np.linalg.norm(velocities[..., :3], axis=1) * 3.6  # m/s -> km/h
                # print("velocity magnitudes: ", velocity_magnitudes)

                overlay_width, overlay_height = bbox_overlay.size
                projected_corners *= np.array([[overlay_width, overlay_height]])

                for i, box in enumerate(projected_corners):
                    velocity_magnitude = velocity_magnitudes[i]
                    if round(velocity_magnitude, 1) > 0:
                        vehicle_front = (points_local[i][2] + points_local[i][3]) / 2
                        vehicle_back = (points_local[i][6] + points_local[i][7]) / 2
                        vehicle_dir = vehicle_front - vehicle_back
                        vehicle_dir /= np.linalg.norm(vehicle_dir)
                        is_intersecting = _ray_sphere_intersection(
                            vehicle_front[:3], vehicle_dir[:3], EGO_INTERSECTION_RADIUS
                        )
                        colour = RED_COLOUR if is_intersecting else GREEN_COLOUR
                        _draw_lollipop(
                            draw, box, distances[i], velocity_magnitude, colour, overlay_width, overlay_height
                        )
                        _draw_arrow(
                            draw,
                            points_local[i],
                            distances[i],
                            velocity_magnitude,
                            view_params,
                            colour,
                            overlay_width,
                            overlay_height,
                        )
                bbox_overlay = _draw_wireframe(
                    bbox_overlay, projected_corners, semantics, distances, velocity_magnitudes, furthest, [200, 0, 0]
                )

    except Exception as error:
        print(f"Exception occurred in save_bbox_and_pc while processing bboxes: {error}")
        traceback.print_exc()

    # Draw the pointcloud into an image.
    pc_overlay = get_point_cloud_overlay(data, view_params, w, h)

    using_vdb = "vdb_points" in data
    if using_vdb:
        pc_overlay = pc_overlay.rotate(180)
    img = Image.alpha_composite(img, pc_overlay)

    # Draw the position of the sensors.
    if "radar" in data:
        sensors_pos_overlay, status = get_sensor_pos_overlay(data, view_params)
        if status:
            img = Image.alpha_composite(img, sensors_pos_overlay)

    # Composite the bounding boxes and the point cloud overlays if available.
    fpath = ""
    if bboxes:
        bbox_overlay = bbox_overlay.resize([w, h])
        img = Image.alpha_composite(img, bbox_overlay)
        fpath = os.path.join(out_dir, "3d_cuboids", get_sensor_name(data), view, "{:04d}{}".format(frame_id, ".png"))
    elif using_vdb:
        fpath = os.path.join(out_dir, "vdb", get_sensor_name(data), view, "{:04d}{}".format(frame_id, ".png"))

    try:
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
    except Exception as error:
        # TODO:  Exception being caused by having 1 more .npy file than ldr_colors. Find out why.
        this_function = inspect.currentframe().f_code.co_name
        print(f"{this_function} error: {error} for frame id: {frame_id}")
        traceback.print_exc()
        return

    arr = np.asarray(img)[..., [2, 1, 0, 3]]
    success, buf = cv2.imencode(".png", arr)
    buf = buf.tobytes()

    backend.write_blob(fpath, buf)


def show_semantic_segmentation(backend, data, out_dir=""):
    # Write semantic in colour
    fpath = os.path.join(out_dir, "semantic_segmentation", get_sensor_name(data), f"{data['frame_id']:04d}.png")
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    # semantic_segmentation = colorize_segmentation(data["semantic_segmentation"])
    semantic_segmentation = colorize_segmentation(
        data["semantic_segmentation"], data["semantic_mapping"], semantic=True
    )
    img = Image.fromarray(semantic_segmentation)

    # buf = io.BytesIO()
    # img.save(buf, format="png", compress_level=1)
    # backend.write_blob(fpath, buf.getvalue())

    # OpenCV encoding is +30% faster
    arr = np.asarray(img)[..., [2, 1, 0]]
    success, buf = cv2.imencode(".png", arr)
    buf = buf.tobytes()
    backend.write_blob(fpath, buf)


def show_instance_segmentation(backend, data, out_dir=""):
    fpath = os.path.join(out_dir, "instance_segmentation", get_sensor_name(data), f"{data['frame_id']:04d}.png")
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    instance_segmentation = colorize_segmentation(data["instance_segmentation"], data["instance_mapping"])
    instance_segmentation[..., -1] = instance_segmentation[..., -1] / 255 * 150
    img = Image.fromarray(instance_segmentation)

    rgb = Image.fromarray(data["rgb"])
    rgb = rgb.convert("RGBA")
    img = Image.alpha_composite(rgb, img)

    # buf = io.BytesIO()
    # img.save(buf, format="png", compress_level=1)
    # backend.write_blob(fpath, buf.getvalue())

    # OpenCV encoding is +30% faster
    arr = np.asarray(img)[..., [2, 1, 0]]
    success, buf = cv2.imencode(".png", arr)
    buf = buf.tobytes()
    backend.write_blob(fpath, buf)
