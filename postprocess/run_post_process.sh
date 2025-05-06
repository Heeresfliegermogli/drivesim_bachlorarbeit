#!/bin/bash

set -e
SCRIPT_DIR=$(dirname ${BASH_SOURCE})

PYTHON=$SCRIPT_DIR/_build/linux-x86_64/release/python.sh
POST_PROC_DIR=$SCRIPT_DIR/source/python/scripts/sdg_post_processing

echo "Running silent pip installs for deps..."
$PYTHON -m pip install -q -r $POST_PROC_DIR/requirements.txt 
echo " "

"$PYTHON" \
"$POST_PROC_DIR/post_process_core.py" $@ || exit $?
