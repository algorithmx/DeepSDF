#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
# Modified Python version that uses trimesh instead of C++ executables

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import threading
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import deep_sdf
import deep_sdf.workspace as ws

# Import the Python preprocessing functions
from preprocess_mesh_python import preprocess_mesh as preprocess_sdf
from sample_visible_mesh_python import (
    sample_visible_mesh_surface,
    save_surface_samples_to_ply,
    save_normalization_params,
)


def filter_classes_glob(patterns, classes):
    import fnmatch

    passed_classes = set()
    for pattern in patterns:
        passed_classes = passed_classes.union(
            set(filter(lambda x: fnmatch.fnmatch(x, pattern), classes))
        )

    return list(passed_classes)


def filter_classes_regex(patterns, classes):
    import re

    passed_classes = set()
    for pattern in patterns:
        regex = re.compile(pattern)
        passed_classes = passed_classes.union(set(filter(regex.match, classes)))

    return list(passed_classes)


def filter_classes(patterns, classes):
    if patterns[0] == "glob":
        return filter_classes_glob(patterns, classes[1:])
    elif patterns[0] == "regex":
        return filter_classes_regex(patterns, classes[1:])
    else:
        return filter_classes_glob(patterns, classes)


def process_mesh_sdf(mesh_filepath, target_filepath, additional_args):
    """Process mesh for SDF sampling using Python implementation."""
    logging.info(mesh_filepath + " --> " + target_filepath)

    # Parse arguments
    num_samples = 500000
    variance = 0.005
    test_flag = False
    sign_method = "vote"

    for i in range(len(additional_args)):
        if additional_args[i] == "-s" and i + 1 < len(additional_args):
            num_samples = int(additional_args[i + 1])
        elif additional_args[i] == "--var" and i + 1 < len(additional_args):
            variance = float(additional_args[i + 1])
        elif additional_args[i] == "-t":
            test_flag = True
        elif additional_args[i] == "--sign-method" and i + 1 < len(additional_args):
            sign_method = additional_args[i + 1]

    return preprocess_sdf(
        mesh_path=mesh_filepath,
        output_path=target_filepath,
        num_sample=num_samples,
        variance=variance,
        test_flag=test_flag,
        save_ply=False,
        sign_method=sign_method,
    )


def process_mesh_surface(mesh_filepath, target_filepath, additional_args):
    """Process mesh for surface sampling using Python implementation."""
    logging.info(mesh_filepath + " --> " + target_filepath)

    # Parse normalization parameter path if provided
    norm_params_path = None
    for i in range(len(additional_args)):
        if additional_args[i] == "-n" and i + 1 < len(additional_args):
            norm_params_path = additional_args[i + 1]
            break

    # Import here to avoid issues if trimesh not installed
    try:
        import trimesh
    except ImportError:
        logging.error("trimesh is required. Install with: pip install trimesh")
        raise

    # Load mesh
    mesh = trimesh.load(mesh_filepath, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            trimesh.Trimesh(**g.geometry) for g in mesh.geometry.values()
        )

    # Compute and apply normalization
    from sample_visible_mesh_python import (
        compute_normalization_parameters,
        normalize_mesh_with_params,
    )

    translation, scale = compute_normalization_parameters(mesh)
    normalize_mesh_with_params(mesh, translation, scale)

    # Sample visible surface
    points, normals = sample_visible_mesh_surface(mesh, num_samples=100000, num_views=100)

    # Save to PLY
    save_surface_samples_to_ply(points, normals, target_filepath)

    # Save normalization parameters if requested
    if norm_params_path:
        save_normalization_params(translation, scale, norm_params_path)


# Global variables for progress tracking
_progress_counter = 0
_progress_total = 0
_progress_lock = threading.Lock()
_progress_start_time = None
_progress_samples_done = 0
_progress_samples_total = 0
_progress_only_mode = False


class _ProgressOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        msg = record.getMessage()
        return msg.startswith("Global progress:") or msg.startswith("Global summary:")


def _process_with_progress(func, mesh_filepath, target_filepath, additional_args):
    """Wrapper function that processes a mesh and updates progress counter."""
    global _progress_counter
    global _progress_total
    global _progress_start_time
    global _progress_lock
    global _progress_samples_done
    global _progress_samples_total
    global _progress_only_mode

    try:
        result = func(mesh_filepath, target_filepath, additional_args)
        samples_written = 0
        if isinstance(result, dict) and "samples" in result:
            samples_written = int(result["samples"])

        with _progress_lock:
            _progress_counter += 1
            _progress_samples_done += samples_written
            elapsed = time.time() - _progress_start_time
            rate = _progress_counter / elapsed if elapsed > 0 else 0
            remaining = (_progress_total - _progress_counter) / rate if rate > 0 else 0

            # Sample-based progress (best-effort). Total is the nominal requested
            # samples-per-mesh * mesh count, not the post-filter retained samples.
            sample_pct = (
                (_progress_samples_done * 100.0 / _progress_samples_total)
                if _progress_samples_total > 0
                else 0.0
            )

            # Log progress every 10 files or for the last file
            if _progress_counter % 10 == 0 or _progress_counter == _progress_total:
                logging.info(
                    "Global progress: "
                    + f"samples {_progress_samples_done}/{_progress_samples_total} "
                    + f"({sample_pct:.1f}%) "
                    + f"- meshes {_progress_counter}/{_progress_total} "
                    + f"({_progress_counter * 100.0 / _progress_total:.1f}%) "
                    + f"- Rate: {rate:.2f} files/sec "
                    + f"- ETA: {remaining:.0f} sec ({remaining/60:.1f} min)"
                )
    except Exception as e:
        logging.error(f"Error processing {mesh_filepath}: {e}")
        raise


def append_data_source_map(data_dir, name, source):

    data_source_map_filename = ws.get_data_source_map_filename(data_dir)

    print("data sources stored to " + data_source_map_filename)

    data_source_map = {}

    if os.path.isfile(data_source_map_filename):
        with open(data_source_map_filename, "r") as f:
            data_source_map = json.load(f)

    if name in data_source_map:
        if not data_source_map[name] == os.path.abspath(source):
            raise RuntimeError(
                "Cannot add data with the same name and a different source."
            )

    else:
        data_source_map[name] = os.path.abspath(source)

        with open(data_source_map_filename, "w") as f:
            json.dump(data_source_map, f, indent=2)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Pre-processes data from a data source (Python version, no C++/Pangolin needed)",
    )
    arg_parser.add_argument(
        "--data_dir",
        "-d",
        dest="data_dir",
        required=True,
        help="The directory which holds all preprocessed data.",
    )
    arg_parser.add_argument(
        "--source",
        "-s",
        dest="source_dir",
        required=True,
        help="The directory which holds the data to preprocess and append.",
    )
    arg_parser.add_argument(
        "--name",
        "-n",
        dest="source_name",
        default=None,
        help="The name to use for the data source. If unspecified, it defaults to the "
        + "directory name.",
    )
    arg_parser.add_argument(
        "--split",
        dest="split_filename",
        required=True,
        help="A split filename defining the shapes to be processed.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        default=False,
        action="store_true",
        help="Deprecated: previously-processed outputs are skipped automatically",
    )
    arg_parser.add_argument(
        "--threads",
        dest="num_threads",
        default=8,
        help="The number of threads to use to process the data.",
    )
    arg_parser.add_argument(
        "--test",
        "-t",
        dest="test_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce SDF samples for testing",
    )
    arg_parser.add_argument(
        "--surface",
        dest="surface_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce mesh surface samples for evaluation. "
        + "Otherwise, the script will produce SDF samples for training.",
    )

    arg_parser.add_argument(
        "--sign-method",
        choices=["vote", "proximity", "igl"],
        default="vote",
        help="SDF sign method (training mode only): 'vote', 'proximity', or 'igl'",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    # Two-level progress reporting:
    # - default: only global progress (samples + meshes)
    # - --debug: keep all existing per-mesh logging
    if not args.debug and not args.quiet:
        _progress_only_mode = True
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.addFilter(_ProgressOnlyFilter())

    additional_general_args = []

    if args.surface_sampling:
        subdir = ws.surface_samples_subdir
        extension = ".ply"
        process_func = process_mesh_surface
    else:
        subdir = ws.sdf_samples_subdir
        extension = ".npz"
        process_func = process_mesh_sdf

        additional_general_args += ["--sign-method", args.sign_method]

        if args.test_sampling:
            additional_general_args += ["-t"]

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    if args.source_name is None:
        args.source_name = os.path.basename(os.path.normpath(args.source_dir))

    dest_dir = os.path.join(args.data_dir, subdir, args.source_name)

    logging.info(
        "Preprocessing data from "
        + args.source_dir
        + " and placing the results in "
        + dest_dir
    )

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    if args.surface_sampling:
        normalization_param_dir = os.path.join(
            args.data_dir, ws.normalization_param_subdir, args.source_name
        )
        if not os.path.isdir(normalization_param_dir):
            os.makedirs(normalization_param_dir)

    append_data_source_map(args.data_dir, args.source_name, args.source_dir)

    class_directories = split[args.source_name]

    meshes_targets_and_specific_args = []

    for class_dir in class_directories:
        class_path = os.path.join(args.source_dir, class_dir)
        instance_dirs = class_directories[class_dir]

        logging.debug(
            "Processing " + str(len(instance_dirs)) + " instances of class " + class_dir
        )

        target_dir = os.path.join(dest_dir, class_dir)

        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        for instance_dir in instance_dirs:

            shape_dir = os.path.join(class_path, instance_dir)

            processed_filepath = os.path.join(target_dir, instance_dir + extension)
            # Skip already-processed outputs based on existence in target folder.
            # This avoids recomputing when re-running the script.
            if os.path.isfile(processed_filepath):
                logging.debug("skipping existing " + processed_filepath)
                continue

            try:
                mesh_filename = deep_sdf.data.find_mesh_in_directory(shape_dir)

                specific_args = []

                if args.surface_sampling:
                    normalization_param_target_dir = os.path.join(
                        normalization_param_dir, class_dir
                    )

                    if not os.path.isdir(normalization_param_target_dir):
                        os.mkdir(normalization_param_target_dir)

                    normalization_param_filename = os.path.join(
                        normalization_param_target_dir, instance_dir + ".npz"
                    )
                    specific_args = ["-n", normalization_param_filename]

                meshes_targets_and_specific_args.append(
                    (
                        mesh_filename,  # Already the full path
                        processed_filepath,
                        specific_args,
                    )
                )

            except deep_sdf.data.NoMeshFileError:
                logging.warning("No mesh found for instance " + instance_dir)
            except deep_sdf.data.MultipleMeshFileError:
                logging.warning("Multiple meshes found for instance " + instance_dir)

    logging.info(f"Processing {len(meshes_targets_and_specific_args)} meshes...")

    # Initialize progress tracking
    _progress_counter = 0
    _progress_total = len(meshes_targets_and_specific_args)
    _progress_start_time = time.time()

    # Nominal total samples is target samples-per-mesh * mesh count.
    # The actual retained sample count can be lower due to filtering.
    _progress_samples_done = 0
    _progress_samples_total = (500000 * _progress_total) if not args.surface_sampling else 0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.num_threads)
    ) as executor:

        for (
            mesh_filepath,
            target_filepath,
            specific_args,
        ) in meshes_targets_and_specific_args:
            executor.submit(
                _process_with_progress,
                process_func,
                mesh_filepath,
                target_filepath,
                specific_args + additional_general_args,
            )

        executor.shutdown()

    elapsed = time.time() - _progress_start_time
    logging.info(
        "Global summary: "
        + f"processed {_progress_counter} meshes in {elapsed:.1f} seconds "
        + f"({elapsed/60:.1f} minutes) at {_progress_counter/elapsed:.2f} files/sec"
    )
