#!/bin/bash

$CMD="$1"

PYTHON="./venv3.11/bin/python3"

$PYTHON -m build

if [ "$CMD" = "upload" ] ; then
    $PYTHON -m twine upload --repository pypi dist/*
fi

exit 0

