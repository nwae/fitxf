#!/bin/bash

CMD="$1"
PKGVER="$2"

PYTHON="./venv3.11/bin/python3"

$PYTHON -m build

if [ "$CMD" = "upload" ] ; then
    if [ "$PKGVER" = "" ] ; then
        echo "Package version not specified"
        exit 1
    fi
    echo "Uploading to pypi pkg version $PKGVER"
    $PYTHON -m twine upload --repository pypi "dist/fitxf-$PKGVER"*
fi

exit 0

