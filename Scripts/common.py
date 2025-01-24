import argparse
import os
import json
from typing import Any

"""
Experiment standard parser.
Reads command line arguments and returns a namespace.
**Does Not Extensively Validate Input** -This is the responsibility of the user
"""


def log(obj: object = ""):
    """
    Logs input to terminal and to log file.
    Handy for debugging.
    """
    string = str(obj)
    print(string)
    with open("log.txt", "a") as log_handle:
        log_handle.write(f"{string}\n")


def is_bool(string: str) -> bool:
    return string.lower() in ["t", "true", "f", "false"]


def is_int(string: str) -> bool:
    try:
        int(string)
        return True
    except ValueError:
        return False


def is_float(string: str) -> bool:
    try:
        float(string)
        return True
    except ValueError:
        return False


def parse_args() -> argparse.Namespace:
    """
    Parses input arguments.
    Validates types.
    Validates replications > 1.
    """
    parser = argparse.ArgumentParser(
        prog="Experiment Parser",
        description="Run a differentially private algorithm on a graph",
    )
    parser.add_argument(
        "-in", "--input-file", required=True, help="The input file path"
    )
    parser.add_argument(
        "-out", "--output-file", required=True, help="The output file path"
    )
    parser.add_argument(
        "-e", "--epsilon", required=True, help="The epsilon privacy budget"
    )
    parser.add_argument(
        "-d", "--delta", required=False, default="0", help="The delta privacy budget"
    )
    parser.add_argument(
        "-r",
        "--replications",
        required=False,
        default="1",
        help="Number of times replicate this experiment",
    )
    parser.add_argument(
        "--disable",
        required=False,
        default="false",
        help="Disables the computation of results",
    )

    log("Read arguments")
    args = parser.parse_args()
    log(f"input_file: {args.input_file}")
    log(f"output_file: = {args.output_file}")
    log(f"epsilon: = {args.epsilon}")
    log(f"delta: = {args.delta}")
    log(f"replications: = {args.replications}")
    log(f"disable: = {args.disable}")
    log("Read arguments - Done")
    log()

    log("Validating arguments")
    if not is_float(args.epsilon):
        error = f"Epsilon {args.epsilon} is not a float"
        log(error)
        raise Exception(error)
    args.epsilon = float(args.epsilon)

    if not is_float(args.delta):
        error = f"Delta {args.delta} is not a float"
        log(error)
        raise Exception(error)
    args.delta = float(args.delta)

    if not is_int(args.replications):
        error = f"Replications {args.replications} is not an int"
        log(error)
        raise Exception(error)
    args.replications = int(args.replications)

    if not is_bool(args.disable):
        error = f"Disable {args.disable} is not an int"
        log(error)
        raise Exception(error)
    args.disable = args.disable.lower() in ["t", "true"]

    if not os.path.exists(args.input_file):
        error = f"File {args.input_file} does not exist"
        log(error)
        raise Exception(error)

    if not os.path.isfile(args.input_file):
        error = f"File {args.input_file} is not a file"
        log(error)
        raise Exception(error)

    if args.epsilon < 0:
        error = f"Epsilon must be at least 0 not {args.epsilon}"
        args.epsilon = 0
        log(error)
        raise Warning(error)

    if args.delta < 0:
        error = f"Delta must be at least 0 not {args.delta}"
        args.delta = 0
        log(error)
        raise Warning(error)

    if args.replications < 1:
        error = (
            f"Replications must be at least 1. To disable this run use --disable=True"
        )
        args.replications = 1
        log(error)
        raise Warning(error)

    log("Validating arguments - Done")
    log()
    return args


def write_results(
    output_file,
    args: argparse.Namespace,
    alert: str | None = None,
    results: list[dict[str, Any]] | None = None,
):
    """
    Write outputs to a json file.

    :param output_file: The file path to where the file is located. Note that directories are created along this path.
    :param args: Arguments passed via command line. Used for debugging, analysis and caching.
    :param alert: Should the reader of this file be alerted to anything, like a disable or an error
    :param results: A list of results of each run. Stored individually as a dictionary.
    :return: None
    """
    meta = {}
    for key in args.__dict__.keys():
        meta[key] = args.__dict__[key]

    output = {}
    output["alert"] = alert
    output["meta"] = meta
    if results != None:
        output["results"] = results

    # Make a path to output file
    target_dir = os.path.dirname(output_file)
    if target_dir != "" and not os.path.exists(target_dir):
        log(f"Creating directory {target_dir}")
        os.makedirs(target_dir, exist_ok=True)
        log(f"Creating directory {target_dir} - Done")

    log(f"Saving results to {output_file}")
    # Use a with-clause to ensure resource safety in the event of an exception
    with open(output_file, "w") as output_handle:
        json.dump(output, output_handle)

    log(f"Saving results to {output_file} - Done")
    log()
