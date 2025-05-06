# Post process GTC output

import argparse
import atexit
import functools
import os
import queue
import sys
import threading

# from pdb import set_trace
import traceback
from importlib import util as imutil

import toml

# Future TODO:
# 1. Benchmark baseline data
# 2. Try integrating pp_queue and backend in processor.self to further parallelize processing


class IOQueue:
    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        atexit.register(self.destroy)
        self._out_dir = out_dir
        num_threads = 20  # TODO: Investigate if raising this increases perf
        self.num_worker_threads = num_threads
        self.q = queue.Queue()
        self.threads = []

    def start_threads(self):
        # Start worker threads
        for _ in range(self.num_worker_threads):
            t = threading.Thread(target=self.worker, daemon=True)
            t.start()
            self.threads.append(t)

    def wait_until_done(self):
        print("Finish writing data...")

        # Block until all tasks are done
        self.q.join()

        print("Done.")

    def destroy(self):
        self.wait_until_done()

        print("Tearing down DataWriteQueue threads...")

        # Stop workers
        for _ in range(self.num_worker_threads):
            self.q.put(None)
        for t in self.threads:
            t.join()

        print("Done tearing down DataWriteQueue threads.")

    def _write_blob(self, identifier_path, blob):
        full_path = self._resolve_path(identifier_path)
        dirname = os.path.dirname(full_path)
        os.makedirs(dirname, exist_ok=True)
        print(f"Writing {full_path}")
        with open(full_path, "wb") as fp:
            fp.write(blob)

    def _resolve_path(self, identifier_path):
        return os.path.join(self._out_dir, identifier_path)

    def _read_blob(self, identifier_path):
        full_path = os.path.join(self._out_dir, identifier_path)
        with open(full_path, "rb") as rb:
            return rb.read()

    def worker(self):
        while True:
            payload = self.q.get()
            if payload is None:
                break
            identifier_path, blob = payload
            try:
                self._write_blob(identifier_path, blob)
            except Exception as e:
                print(e)
                traceback.print_exc()
            self.q.task_done()

    def write_blob(self, identifier_path, blob):
        self.q.put((identifier_path, blob))


class PostProcessQueue:
    def __init__(self):
        atexit.register(self.destroy)
        num_threads = 20
        self.num_worker_threads = num_threads
        self.q = queue.Queue()
        self.threads = []

    def start_threads(self):
        # Start worker threads
        for _ in range(self.num_worker_threads):
            t = threading.Thread(target=self.worker, daemon=True)
            t.start()
            self.threads.append(t)

    def wait_until_done(self):
        print("Finish writing data...")

        # Block until all tasks are done
        self.q.join()

        print("Done.")

    def destroy(self):
        self.wait_until_done()

        print("Tearing down DataWriteQueue threads...")

        # Stop workers
        for _ in range(self.num_worker_threads):
            self.q.put(None)
        for t in self.threads:
            t.join()

        print("Done tearing down DataWriteQueue threads.")

    def worker(self):
        while True:
            fn = self.q.get()
            if fn is None:
                break
            try:
                fn()
            except Exception as e:
                partial_fn = functools.partial(fn, 1)
                print(f"Exception: Occurred while executing {partial_fn.func.__name__} thread function: {e}")
                traceback.print_exc()
            self.q.task_done()

    def visualize(self, fn):
        self.q.put(fn)


class DiskBackend:
    def __init__(self, out_dir):
        self._out_dir = out_dir
        print(f"Writing data to {self._out_dir}")

    def write_blob(self, identifier_path, blob):
        full_path = self.resolve_path(identifier_path)
        dirname = os.path.dirname(full_path)
        os.makedirs(dirname, exist_ok=True)
        print(f"Writing {full_path}")
        with open(full_path, "wb") as fp:
            fp.write(blob)


def try_get_pprocessor_src(writer: str):
    # Unknown if there's any better way to get the project root outside of Kit
    # Docker will not have source/extensions, only _build, so search that.
    exts_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../_build/linux-x86_64/release/exts/")
    )
    gen_extname = "omni.drivesim.replicator.writers/"
    int_extname = "omni.drivesim.datastudio.driveav.writers/"
    ext_names = [gen_extname, int_extname]
    ext_paths = [os.path.join(exts_root, gen_extname), os.path.join(exts_root, int_extname)]
    writers_found = []
    # Build list of available writers
    for en, ep in zip(ext_names, ext_paths):
        conf_p = os.path.join(ep, "config/extension.toml")
        src_path = os.path.join(ep, en.replace(".", "/"))
        if not os.path.exists(conf_p):
            print(f"Searched extension config path '{conf_p}', but it does not exist!")
            continue
        if not os.path.exists(src_path):
            print(f"Searched writer source path '{src_path}', but it does not exist!")
            continue
        with open(conf_p, "r") as f:
            writer_paths = toml.load(f).get("replicator_writers")
            if writer_paths is not None:
                for w, p in writer_paths.items():
                    writers_found.append(w)
                    if w == writer:
                        return writers_found, os.path.join(src_path, p)
    return writers_found, ""


def get_available_writer_names():
    msg = "The following writers are available with drivesim's post processing: \n"
    writers_found, _ = try_get_pprocessor_src("")
    for w in writers_found:
        msg += f" - {w} \n"

    msg += (
        "The following writers are available with voxel 51\n - BasicDSWriterFull \n - PointCloudWriter (with lidar) \n"
    )
    return msg


def main(source, output_dir, writer, **kwargs):
    processor = None
    if writer is None:
        print("Writer name not specified! Script cannot run!")
        print("Please specify a valid writer.")
        print(get_available_writer_names())
        exit(1)
    writers_found, pproc_src = try_get_pprocessor_src(writer)
    if pproc_src == "":
        print(f"Source for writer {writer} not detected! Script cannot run!")
        print(f"Found writers: {writers_found}")
        exit(1)
    # Now given the writer name and post processing source, dynamically load class from module
    try:
        module_name = pproc_src.split("/")[-1].replace(".py", "")
        writer_spec = imutil.spec_from_file_location(module_name, pproc_src)
        writer_mod = imutil.module_from_spec(writer_spec)
        sys.modules[module_name] = writer_mod
        writer_spec.loader.exec_module(writer_mod)
        processor = writer_mod.PostProcessor
    except ImportError as e:
        print(f"Failed to import post-processing module or class for {writer}!")
        print(f"Received exception: {e}")
        traceback.print_exc()
        exit(1)
    print(f"Writer {writer} imported, loading PostProcessor and starting jobs...\n")
    pp_queue = PostProcessQueue()
    # Name output subdir as the 'src_dir_name_processed' to allow easy overwrite and tracking
    src_dir_name = source.split("/")[-1]
    backend = IOQueue(os.path.join(output_dir, f"{src_dir_name}_processed"))
    pp_queue.start_threads()
    backend.start_threads()
    processor.run_post_process(source, pp_queue, backend, **kwargs)
    pp_queue.wait_until_done()
    backend.wait_until_done()
    if kwargs["create_video"] and not kwargs["frame_sel"]:
        print("Video creation requested! Calling writer's video method now...")
        framerate = kwargs.pop("framerate")
        processor.do_create_video(os.path.abspath(backend._out_dir), framerate=framerate, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="SDG Post-processing script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=get_available_writer_names(),
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        help="Source directory (e.g. /tmp/replicator_out/BasicDSWriterFull_out/scenario_entry_timestamp",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        help="Output directory (e.g. /tmp/replicator_out/BasicDSWriterFull_out/scenario_viz",
    )
    parser.add_argument("--writer", "-w", type=str, help="Writer name (e.g. BasicDSWriterFull)")
    parser.add_argument("--frame", "-f", type=int, help="(Optional) only visualize one frame")
    parser.add_argument("--session", type=str, help="(Optional) only visualize one session")
    parser.add_argument("--camera", "-c", type=str, help="(Optional) only visualize one camera")
    parser.add_argument("--profile", action="store_true", help="Profile code using pyinstrument")
    parser.add_argument("--debug", action="store_true", help="Run post-processing synchronously to aid in debugging")
    parser.add_argument("--create_video", action="store_true", help="Create video from the frames")
    parser.add_argument("--lidar_only", action="store_true", help="Generate frames for lidar only")
    parser.add_argument(
        "--view",
        type=str,
        default="third_person",
        help="[Lidar/Radar only] Select the perspective from which to view the lidar points. \
            Choose from [first_person, top_down, third_person]",
    )
    parser.add_argument("--color", choices=["red", "green", "blue"], help="Choose a color for the point cloud.")
    parser.add_argument("--framerate", type=int, help="Framerate for video creation")

    # For voxel51
    parser.add_argument(
        "--fiftyone", action="store_true", help="Post process data to use with Voxel51 instead of Replicator Insight"
    )

    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    args, unknown_args = parser.parse_known_args()

    if args.lidar_only and args.framerate is None:
        args.framerate = 10
    else:
        args.framerate = 30

    # To be compatible with the legacy shell script version arg syntax
    positional_args = [parg for parg in unknown_args if not parg.startswith("--")]
    unknown_flag_args = [uarg for uarg in unknown_args if uarg.startswith("--")]
    if len(positional_args) > 3:
        print("Received too many unspecified args! See usage for positional args or specify flags.")
        parser.print_help()
        exit(1)
    elif len(unknown_flag_args) > 0:
        print("Found the following invalid keyword --args:")
        for a in unknown_flag_args:
            print(f"  {a}")
        print("Exiting the script, please see --help menu for proper usage.")
        exit(1)
    else:
        for i, p in enumerate(positional_args):
            if len(p) < 1:
                print(f"Received bad arg '{p}'! Exiting...")
                exit(1)
            if i == 0:
                args.source = os.path.abspath(p)
            elif i == 1:
                args.output_dir = p
            elif i == 2:
                args.writer = p

    if not os.path.exists(args.source):
        print(f"Input directory not found! Could not find: '{args.source}'")
        print("Exiting...")
        exit(1)

    if args.fiftyone:
        from voxel51.fiftyone_core import main as fo_main

        print("Initializing SDG post processing for Voxel51. All other arguments except input directory are ignored")
        fo_main(data_path=args.source, writer=args.writer)
    else:
        if args.profile:
            import tracemalloc

            from pyinstrument import Profiler

            profiler = Profiler()
            tracemalloc.start()
            profiler.start()

        print("Initializing SDG Post-processing script...")
        main(
            args.source,
            args.output_dir,
            args.writer,
            frame_sel=args.frame,
            camera_sel=args.camera,
            session_sel=args.session,
            create_video=args.create_video,
            lidar_only=args.lidar_only,
            color=args.color,
            view=args.view,
            debug=args.debug,
            framerate=args.framerate,
        )

        if args.profile:
            profiler.stop()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            current = current / (1024 * 1024)
            peak = peak / (1024 * 1024)
            profiler.print()
            print("Current and peak memory usage [MB]: {} {}".format(current, peak))
