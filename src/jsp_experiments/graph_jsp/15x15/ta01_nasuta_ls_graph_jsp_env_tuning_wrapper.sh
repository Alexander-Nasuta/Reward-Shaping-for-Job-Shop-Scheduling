#!/bin/bash
# make sure you have an env named jsp setup with all requirements installed
source activate jsp
for i in {1..50}; do python ta01_nasuta_ls_graph_jsp_env_tuning.py; done