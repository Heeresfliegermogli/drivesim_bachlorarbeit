#!/bin/bash

set -e
SCRIPT_DIR=$(dirname ${BASH_SOURCE})

PYTHON=$SCRIPT_DIR/_build/linux-x86_64/release/python.sh

"$SCRIPT_DIR/tools/packman/python.sh" "$SCRIPT_DIR/generate_SDG_headless.py" $@ || exit $?
