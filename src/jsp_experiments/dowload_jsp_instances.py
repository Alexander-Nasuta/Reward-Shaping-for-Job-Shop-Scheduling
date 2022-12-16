import os
import pathlib as pl
import graph_jsp_utils.jsp_instance_downloader as jsp_downloader

from graph_jsp_env.disjunctive_graph_logger import log

if __name__ == '__main__':
    log("downloading all available jsp instances.")
    target_dir = pl.Path(os.path.abspath(__file__)) \
        .parent \
        .parent \
        .parent \
        .joinpath("resources") \
        .joinpath("jsp_instances")

    jsp_downloader.download_instances(
        target_directory=target_dir
    )
