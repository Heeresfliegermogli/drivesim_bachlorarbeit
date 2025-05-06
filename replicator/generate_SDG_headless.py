__copyright__ = "Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

# flake8: noqa

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from generate_SDG_headless_args import add_replicator_args


class SDGHeadless:
    def __init__(self, additional_scenarios=None):
        self.scenarios = self.default_scenarios
        if additional_scenarios:
            self.scenarios.update(additional_scenarios)
        self.run_dir = ""

    @property
    def default_scenarios(self):
        return {
            "sdg_pyds_loopmerge": {
                "path": "./tools/py_drivesim2/integration_tests/sdg_pydrivesim_loopmerge.py",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=pyds_loop.toml",
            },
            "multi_maps": {
                "path": "./assets/scenarios/sdg_map_list.txt",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=ego_config.toml",
            },
            "seq_multi_maps": {
                "path": "./assets/scenarios/sdg_map_list.txt",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=sequential_example_dynamic_ego_minimal.toml",
            },
            "rivermark": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/rivermark_parking_lot.usda",  # TODO: Remove 'moveTo' action
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=randomizer_config.toml",
            },
            "rivermark_peds": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/rivermark_parking_lot.usda",  # TODO: Remove 'moveTo' action
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_dir=${omni.drivesim.replicator.domainrand}/omni/drivesim/replicator/domainrand/config/test_configs --/omni/drivesim/replicator/dr/config_name=rivermark_peds_test.toml",
            },
            "signs": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/low_res/poc_ottosuhrallee_1080p.usda",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=signs_config.toml",
            },
            "seq_signs": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/low_res/poc_ottosuhrallee_1080p.usda",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=sequential_example.toml",
            },
            "seq_bikes": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/low_res/poc_ottosuhrallee_1080p.usda",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=sequential_bikes_example.toml",
            },
            "seq_rivermark": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/poc_rivermark_official_peds-rel.usda",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=sequential_example.toml",
            },
            "seq_rivermark_ego_spawned": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/poc_rivermark_peds_no_ego.usda",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=sequential_example_dynamic_ego.toml",
                "other_flags": "--/omni/drivesim/replicator/egoPrimPath=/Entities/Ego_sdg",
            },
            "pedgate_hamburg": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/hamburg_intersect_1cam_pedgate.usda",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=sequential_pedestrian_example.toml",
            },
            "cones": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/poc_rivermark_official_peds-rel.usda",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=cones_config.toml",
            },
            "weather_example_low_res": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/low_res/poc_ottosuhrallee_1080p.usda",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=weather_ex.toml",
            },
            "weather_example": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/low_res/poc_ottosuhrallee_1080p.usda",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=weather_ex.toml",
            },
            "night_rand": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/low_res/poc_ottosuhrallee_1080p.usda",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=night_rand.toml",
                "render_mode": "--/rtx/rendermode='PathTracing' --/rtx/pathtracing/clampSpp=0 --/rtx/pathtracing/spp=32",
            },
            "seq_night": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/low_res/poc_ottosuhrallee_1080p.usda",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=sequential_night.toml",
                "render_mode": "--/rtx/rendermode='PathTracing' --/rtx/pathtracing/clampSpp=0 --/rtx/pathtracing/spp=32",
            },
            "generic_lidar": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/low_res/loop_generic_lidar.usda",
                "writer": "PointCloudWriter:run_post_processing=True;view=top_down",
                "lockstep_flags": "--/app/drivesim/lockstep/enabled=true  --/app/drivesim/throttleLockstep=true --/app/drivesim/lockstep/useNonLockstepProtocol=true --/omni/replicator/debug=true",
                "raytracing_motion_flags": "--/renderer/raytracingMotion/enabled=true",
                "bin_recoder_flags": "--merge-config=assets/scenarios/recording_generic.toml",
            },
            "generic_lidar_2": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/low_res/sdg_test_scenario_5peds_low_res_rotary_generic.usda",
                "writer": "PointCloudWriter:run_post_processing=True;view=top_down",
                "lockstep_flags": "--/app/drivesim/lockstep/enabled=true  --/app/drivesim/throttleLockstep=true --/app/drivesim/lockstep/useNonLockstepProtocol=true --/omni/replicator/debug=true",
                "raytracing_motion_flags": "--/renderer/raytracingMotion/enabled=true",
                "bin_recoder_flags": "--merge-config=assets/scenarios/recording_generic.toml",
                "label_flags": "--/omni/replicator/user_labels=assets/scenarios/label_configs/loop_pointcloud_labels.toml",
            },
            "generic_radar": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/low_res/loop_radar.usda",
                "writer": "PointCloudWriter:is_radar=true;run_post_processing=True;view=top_down",
                "lockstep_flags": "--/app/drivesim/lockstep/enabled=true  --/app/drivesim/throttleLockstep=true --/app/drivesim/lockstep/useNonLockstepProtocol=true --/omni/replicator/debug=true",
                "raytracing_motion_flags": "--/renderer/raytracingMotion/enabled=true",
                "bin_recoder_flags": "--merge-config=assets/scenarios/recording_generic.toml",
                "label_flags": "--/omni/replicator/user_labels=assets/scenarios/label_configs/loop_pointcloud_labels.toml",
            },
            "slope_parking": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/sdg_slope_parking_1cam.usda",  # Uses vehicle moveTo
                "writer": "BasicDSWriterFull:parking_spaces=True",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=parking_config.toml",
            },
            "intersection_lane": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/hamburg_intersect_1cam.usda",
                "writer": "BasicDSWriterFull:road_lanes=True",
            },
            "loop_merge_ego_spawned": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/loop_merge_no_ego.usda",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=ego_config.toml",
                "other_flags": "--/omni/drivesim/replicator/egoPrimPath=/Entities/Ego_sdg",
            },
            "loop_merge_seq_spawned": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/loop_merge_no_ego.usda",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=sequential_example_dynamic_ego.toml",
                "other_flags": "--/omni/drivesim/replicator/egoPrimPath=/Entities/Ego_sdg",
            },
            "cloud_ego_spawned": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/herrenberg_no_ego.usda",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=ego_config.toml",
                "other_flags": "--/omni/drivesim/replicator/egoPrimPath=/Entities/Ego_sdg",
            },
            "hub_signs": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/low_res/poc_ottosuhrallee_1080p.usda",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=signs_config.toml",
                "writer": "HubWriter",
                "hub_settings": "--/poc/sdg/storageid=/tmp/replicator_out --/poc/sdg/hubsetname=HubWriter_out",
            },
            # Original scene_bends modified to illustrate camera post processing.
            "scene_bends_ccm_exp_example": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/low_res/scene_bends_ccm_exposure.usda",
                "writer": "BasicDSWriterFull",
                "examples_flags": "--enable omni.graph.examples.cpp",
            },
            # New scene bends, which now has itw own HDR RenderVar (as opposed to the basic ldr_color )
            "scene_bends_ccm_exp_hdr": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/scene_bends_for_ccm_exp.usda",
                # RGB output can be one of: ldr_color (default)|hdr_color|hdr_raw_color (requested by customer)
                "writer": "BasicDSWriterFull:hdr_raw_color=True",
            },
            "hello_world": {
                "path": "{/app/drivesim/defaultNucleusRoot}/Projects/ds2_scenarios/sdg/preset_scenes/sample_oval_no_ego.usda",
                "randomizer_flags": "--/omni/drivesim/replicator/dr/config_name=ego_config.toml",
                "other_flags": "--/omni/drivesim/replicator/egoPrimPath=/Entities/Ego_sdg",
            },
            # basic writer oval to test NeRF materials
            "basic_oval": {
                "path": "./assets/scenarios/oval_non_visual.usda",
                "writer": "BasicDSWriterFull",
            },
        }

    @staticmethod
    def sanitize_path(path: str) -> str:
        if not path:
            raise ValueError(f"Invalid path provided: {path}")

        return str(Path(path).expanduser())

    @property
    def default_values(self):
        return {"num_frames": 100, "writer": "BasicDSWriterFull", "simulation_duration": 3}

    @staticmethod
    def set_smart_display_env():
        # do nothing for windows platform
        if sys.platform == "win32":
            return

        if "DISPLAY" in os.environ:
            # do nothing if DISPLAY already set
            pass
        elif os.path.isfile("/.dockerenv"):
            # if we are within docker, do not set DISPLAY variable
            pass
        else:
            # non container environment, use `w -oush` to determine a physical screen
            display = os.popen('''w -oush | awk '{print $2}' | egrep ":[0-9]*"''').read().strip()
            # check if r is valid DISPLAY format
            r = re.match("^:[0-9]*$", display)
            if r is None:
                # not match, do not set
                pass
            else:
                os.environ["DISPLAY"] = display

        # allow request from different terminal to avoid permission issues
        if "DISPLAY" in os.environ:
            os.system("xhost +")

    def print_usage(self, parser):
        scenario_shortcuts = " ".join(list(self.scenarios.keys()))
        usage = (
            f"\n"
            f"Syntax: $PYTHON {sys.argv[0]} <scenario_path_or_shortcut> [number of frames]\n\n"
            f"  Scenario path or shortcut name must be provided, as well as the number of frames desired.\n"
            f"  Default writer, render level, and other parameters can be modified with respective --flag.\n"
            f"  Example: {sys.argv[0]} signs 100 --writer BasicDSWriterFull --render-level medium\n"
            f"  Render level will default to 'ultralow', selecting RayTracedLighting (RTX).\n"
            f"  Supported render level presets are: ultralow (RTX), low (PTX spp=1), medium (PTX spp=16), high (PTX spp=32), ultra (PTX spp=48)"
            f"  Supported writers are: BasicDSWriterFull (default), KittiWriter, BasicDSWriterCloud, and PointCloudWriter\n"
            f"  Supported scenario shortcuts are: {scenario_shortcuts}\n"
            f"  You may also specify scenarios with direct paths to .usda or py_drivesim2 python scripts.\n"
        )

        print(usage)
        parser.print_help()

    def print_flags(self, flags, flag_type="args"):
        print(f"Received {flag_type}:")
        print(flags)

    def print_preset_scenes(self):
        presets = self.default_scenarios
        print("Available scene presets are:\n")
        for k, v in presets.items():
            path = v.get("path", "PATH NOT FOUND")
            print(f"Preset name: {k} \nPath: {path}")

    def get_command_and_args(self, app_kit_path="apps/omni.drivesim.datastudio.kit", parse_cb=None):
        parser = argparse.ArgumentParser()

        parser = add_replicator_args(parser)

        args, unknown_args = parser.parse_known_args()
        kit_extension_sdk_path = f"_build/target-deps/kit_sdk_{args.kit_config}/"
        if sys.platform == "win32":
            kit_exec = "kit.exe"
            platform = "windows-x86_64"
        else:
            kit_exec = "kit"
            platform = "linux-x86_64"

        # To allow for easy specification of required scenario and num_frames positional args
        positional_args = [arg for arg in unknown_args if not arg.startswith("--")]
        presumed_kit_args = [arg for arg in unknown_args if arg not in positional_args]
        print(f"generate_SDG_headless.py received following positional args: {positional_args}")
        print(
            f"generate_SDG_headless.py interpreting following args as kit flags to append to launch command: {presumed_kit_args}"
        )
        num_positional_args = len(positional_args)

        if args.list_scenes:
            self.print_preset_scenes()
            exit(0)

        # Set main args and / or legacy args
        scene_name = args.scenario
        if scene_name is None and num_positional_args > 0 and positional_args[0]:
            scene_name = positional_args[0]
        if scene_name is None:
            self.print_usage(parser)
            self.print_flags(args)
            exit(-1)
        # Only scene name and num frames supported as positional
        if num_positional_args > 2:
            print("\nERROR! Too many positional args provided.")
            self.print_flags(positional_args, "positional args")
            exit(-1)

        # If scene arg is a preset, load the path, otherwise set path to name to support inline USD, USDA or PYDS scenarios
        scene_path = self.scenarios[scene_name]["path"] if scene_name in self.scenarios else scene_name

        if ".txt" not in scene_path and ".usd" not in scene_path and ".py" not in scene_path:
            print(
                f"FAILED TO FIND VALID SCENE! Received path '{scene_path}'.\nAre you using correct scene name? Rerun command with '-l' to see supported presets."
            )
            exit(-1)

        if args.num_frames == 0:
            args.num_frames = (
                positional_args[1]
                if num_positional_args > 1 and int(positional_args[1]) > 0
                else self.default_values["num_frames"]
            )

        # TODO: Pull this into its own sub method
        render_mode = ""
        presumed_render_flags = [
            arg for arg in presumed_kit_args if "rendermode" in arg or "spp" in arg or "pathtracing" in arg
        ]
        # If kit args specify render settings, do not populate render_mode var. Instead these will be passed in with extra args.
        if len(presumed_render_flags) == 0:
            if args.render_level == "ultralow":
                render_mode = '--/rtx/rendermode="RayTracedLighting"'
            elif args.render_level == "low":
                sample_per_pixel = 1
                render_mode = f'--/rtx/rendermode="PathTracing" --/rtx/pathtracing/clampSpp=0 --/rtx/pathtracing/spp={sample_per_pixel}'
            elif args.render_level == "medium":
                sample_per_pixel = 16
                render_mode = f'--/rtx/rendermode="PathTracing" --/rtx/pathtracing/clampSpp=0 --/rtx/pathtracing/spp={sample_per_pixel}'
            elif args.render_level == "high":
                sample_per_pixel = 32
                render_mode = f'--/rtx/rendermode="PathTracing" --/rtx/pathtracing/clampSpp=0 --/rtx/pathtracing/spp={sample_per_pixel}'
            elif args.render_level == "ultra":
                sample_per_pixel = 48
                render_mode = f'--/rtx/rendermode="PathTracing" --/rtx/pathtracing/clampSpp=0 --/rtx/pathtracing/spp={sample_per_pixel}'
            else:
                print(
                    "Invalid render mode, please select from supported modes [ ultralow, low, medium, high, ultra]! Exiting..."
                )
                exit(-1)
        else:
            print("Presumed kit args contained render settings! Skipping default render-level settings...")
            # Check presumed render flags to infer rendermode.
            rt_mode_found = any(a for a in presumed_render_flags if "--/rtx/rendermode=RayTracedLighting" in a)
            pt_mode_found = any(a for a in presumed_render_flags if "--/rtx/rendermode=PathTracing" in a)
            pt_args = [a for a in presumed_render_flags if "pathtracing" in a]
            # We have PT args but no rendermode specified, assume PT.
            if not pt_mode_found and not rt_mode_found and len(pt_args) > 0:
                print(
                    "\nReceived command-line level render args including SPP, but no '/rtx/rendermode' flag found! Forcing PathTracing..."
                )
                presumed_kit_args.append('--/rtx/rendermode="PathTracing"')
            elif rt_mode_found and len(pt_args) > 0:
                print(
                    "\nError! Found RT mode specified in presumed kit rendering flags, but other pathtracing flags were found. Please review for correctness!"
                )
                self.print_flags(presumed_render_flags, "render flags")
                exit(-1)

        # Start building Kit Command
        kit_extension_sdk_path = SDGHeadless.sanitize_path(
            os.path.abspath(os.path.join(kit_extension_sdk_path, kit_exec))
        )
        app_kit_path = SDGHeadless.sanitize_path(
            os.path.abspath(os.path.join(f"_build/{platform}/{args.ds_config}", app_kit_path))
        )
        self.run_dir = f"_build/{platform}/{args.ds_config}/"
        command = [
            kit_extension_sdk_path,
            app_kit_path,
        ]

        # We have four types of scene inputs possible: usda scenarios, pydrivesim scripts, multi-map maplists, and usd maps.
        # We should filter by filetype to know which of these is being targeted.

        # convert to absolute path for locally hosted files
        if os.path.exists(scene_path):
            scene_path = os.path.abspath(scene_path)

        if scene_path.endswith(".usda") or scene_path.endswith(".usd"):
            autotest_arg = (
                f"--ds_runtime_autotest=autotest_sdg.py {scene_path} {args.num_frames}"
                if not args.disable_autotest
                else ""
            )
        elif scene_path.endswith(".py"):
            autotest_arg = (
                f"--ds_runtime_autotest=autotest_py_drivesim_sdg.py {scene_path} {args.num_frames}"
                if not args.disable_autotest
                else ""
            )
        elif scene_path.endswith(".txt"):
            autotest_arg = f"--ds_runtime_autotest=autotest_sdg_multiple_maps.py {scene_path} {args.num_frames}"
            if args.disable_autotest:
                print(f"Error encountered! Disable autotest not supported for multi-map use case. Ignoring flag...")
        else:
            print(f"Error encountered! Received file {scene_path} is of unsupported type!")
            exit(-1)

        if args.manual_play:
            autotest_arg += " --manual-play" if not args.disable_autotest else ""
            args.show_ui = True  # Force disable no-window arg in manual play mode

        command.append(autotest_arg)

        # Logging args
        if args.log_file is not None:
            log_args = [
                "--/log/outputStreamLevel=warning",
                f"--/log/file='{args.log_file.absolute()}'",
                "--/log/level=info",
            ]
            command.extend(log_args)

        # Render args
        rendering_args = (
            f"--/renderer/multiGpu/enabled=true --/rendergraph/asyncEndFrameSubmit=false "
            f"--/rtx/descriptorSets=30000 "
            f"--/rtx/sceneDb/maxInstances=2000000 "
            f"--/rtx/pathtracing/lightcache/cached/alwaysReuse=true "  # fix blurry trees, fireflies DRIVE-13346
            f"{render_mode}"
        )
        command.extend(rendering_args.split())

        realm_args = f"--/realm/hostAlloc=8192m " f"--/realm/deviceAlloc=4096m "
        command.extend(realm_args.split())

        if args.texture_streaming:
            cloud_settings = (
                "--/rtx-transient/resourcemanager/enableTextureStreaming=true --/rtx-transient/resourcemanager/texturestreaming/async=false "
                "--/rtx-transient/resourcemanager/texturestreaming/streamingBudgetMB=0 --/rtx-transient/resourcemanager/texturestreaming/evictionFrameLatency=0 "
                "--/rtx-transient/samplerFeedbackTileSize=0 "
                "--/rtx-transient/resourcemanager/texturestreaming/memoryBudget=0.1"  # restrict percentage of GPU memory used for texture streaming
            )
            command.extend(cloud_settings.split())

        if not args.no_livestream:
            command.append("--enable omni.kit.livestream.native")

        randomizer_flags = ""
        if scene_name in self.scenarios and "randomizer_flags" in self.scenarios[scene_name]:
            randomizer_flags = self.scenarios[scene_name]["randomizer_flags"]

        if args.rand_config is not None:
            randomizer_flags += f" --/omni/drivesim/replicator/dr/config_name={args.rand_config}"
            if args.rand_config_dir is not None:
                randomizer_flags += f" --/omni/drivesim/replicator/dr/config_dir={args.rand_config_dir}"
            if args.rand_asset_dir is not None:
                randomizer_flags += f" --/omni/drivesim/replicator/dr/asset_dir={args.rand_asset_dir}"

        extra_flags = ""
        if scene_name in self.scenarios:
            scene_attributes = self.scenarios[scene_name]
            for key, value in scene_attributes.items():
                if key == "writer":
                    args.writer = value
                elif key != "path" and key != "randomizer_flags" and key != "multi_maps":
                    extra_flags += f"{value} "

        if parse_cb:
            parse_cb(args.writer)

        output_flag = (
            f"--/omni/replicator/backends/disk/outputDir={os.path.abspath(args.output_path)}"
            if args.output_path
            else ""
        )
        if args.allow_root:
            extra_flags += "--allow-root "

        if not args.show_ui:
            extra_flags += "--no-window "

        if args.nucleus_path:
            print("Using this nucleus path: " + args.nucleus_path)
            extra_flags += f"--/app/drivesim/defaultNucleusRoot={args.nucleus_path} "

        # persistent flag that disables HUD
        extra_flags += "--/persistent/app/viewport/displayOptions=0 "
        # persistent flag that hides timeline grid
        extra_flags += "--/persistent/app/viewport/grid.enabled=false "

        # in case users specify nucleus path at docker launch level instead
        if "SDG_EXTRA_ARGS" in os.environ:
            extra_flags += os.environ["SDG_EXTRA_ARGS"]

        if args.custom_env_setting:
            # do nothing if setting not provided
            if args.custom_env_setting is not None:
                if args.custom_env_setting == "omni_env_settings.json":
                    # use default $HOME/omni_env_settings.json if single parameter not provided
                    setting_file = os.path.join(os.environ["HOME"], "omni_env_settings.json")
                else:
                    # use proivided single parameter as config file
                    setting_file = args.custom_env_setting

                print(f"Loading custom env setting from:  {setting_file}")

                with open(setting_file, "r") as f:
                    env_dict = json.load(f)
                    for key in env_dict:
                        value = env_dict[key]
                        if isinstance(value, str):
                            os.environ[key] = value

        if args.smart_display_env:
            self.set_smart_display_env()

        if scene_name in self.scenarios and "examples_flags" in self.scenarios[scene_name]:
            extra_flags += self.scenarios[scene_name]["examples_flags"]

        datastudio_args = (
            "--/datastudio/debug=true "
            "--/datastudio/gen_video=false "
            f"--/omni/replicator/writers={args.writer} "
            f"{randomizer_flags} {output_flag} {extra_flags}"
        ).split()

        # Append scene to replicator unless pydrivesim
        if not scene_path.endswith(".py"):
            datastudio_args.append(f"--/omni/replicator/scene={scene_path}")
        command.extend(datastudio_args)

        # TODO: Eliminate this section
        if args.usd_teleportation:
            # If using USD, turn off USD Monitoring and add HACK for wakeup frames
            command.append(f"--/app/drivesim/monitorUsdWhileRunning={args.monitor_usd}")
            # HACK DRIVE-11521, needed to WAR incorrect bboxes on USD teleported objects
            command.append("--/app/drivesim/wakeupFramesOffset=0")

        if args.profile:
            command.extend(
                "--/profiler/enabled=true  --/app/profilerBackend=tracy --/app/profileFromStart=true --/profiler/gpu=true --/profiler/gpu/tracyinject/enabled=true".split()
            )

        command.extend(presumed_kit_args)

        if args.seed_value:
            command.append(f"--/omni/drivesim/replicator/dr/rand_seed_value={args.seed_value}")

        if args.run_count <= 0:
            print(f"Received invalid run count {args.run_count}! Resetting to '1'.")
            args.run_count = 1
        elif args.run_count > 1:
            print(f"Received arg count {args.run_count}")

        # Last step before flattening command, add new arguments above here
        if args.print_command:
            print("\nRunning command:")
            print(" ".join(command))
        elif args.print_command_only:
            print("\nPrint-only requested. Command built as:")
            print(" ".join(command))
            return exit(0)
        return command, args

    def run(self, command, args):
        if not command:
            return
        os.chdir(self.run_dir)
        quit_received = False
        print(f"***STARTING HEADLESS FRAME GENERATION FOR {args.run_count} LOOPS***\n")
        for loop_num in range(args.run_count):
            if quit_received:
                break
            print(f"STARTING HEADLESS FRAME GENERATION FOR LOOP #: {loop_num+1}\n")
            kit_proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            while not quit_received:
                try:
                    # Pass output to terminal
                    nextline_kit = kit_proc.stdout.readline()
                    if nextline_kit != b"":
                        print(nextline_kit.decode("utf-8").rstrip("\n"))
                    elif kit_proc.poll() is not None:
                        break
                except KeyboardInterrupt:
                    print("\n CTRL+C received! Terminating frame generation loop...")
                    quit_received = True
            _output, errors = kit_proc.communicate()
            if quit_received:
                print("SDG subprocess cancelled by user interrupt!")
            elif kit_proc.returncode == 0:
                print("SDG subprocess complete!")
            else:
                if errors is not None:
                    print(errors)
                raise ChildProcessError("Error running SDG process!")


if __name__ == "__main__":
    sdg_headless = SDGHeadless()
    command, args = sdg_headless.get_command_and_args()
    sdg_headless.print_flags(command, "launch args")
    sdg_headless.run(command, args)
