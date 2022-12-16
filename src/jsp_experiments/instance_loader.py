import os

import numpy as np
import pathlib as pl

import graph_jsp_utils.jsp_instance_parser as parser
from graph_jsp_env.disjunctive_graph_logger import log


def get_instance_std_path_by_name(name: str) -> pl.Path:
    return pl.Path(os.path.abspath(__file__)) \
        .parent \
        .parent \
        .parent \
        .joinpath("resources") \
        .joinpath("jsp_instances") \
        .joinpath("standard") \
        .joinpath(f"{name}.txt")


def get_instance_by_name_as_numpy_array(name: str) -> np.ndarray:
    ta_instance_dir = pl.Path(os.path.abspath(__file__)) \
        .parent \
        .parent \
        .parent \
        .joinpath("resources") \
        .joinpath("jsp_instances") \
        .joinpath("taillard") \
        .joinpath(f"{name}.txt")
    try:
        jsp, _ = parser.parse_jps_taillard_specification(instance_path=ta_instance_dir)
        return jsp
    except FileNotFoundError as err:
        log.error(
            f"could not find any instance with name {name}. Make sure you have downloaded the instance (use the "
            f"'instance_loader.py' script for that). Also make sure the instance name is spelled correctly."
        )
        raise err


if __name__ == '__main__':
    instance = get_instance_by_name_as_numpy_array("ft06")
    print(instance)
