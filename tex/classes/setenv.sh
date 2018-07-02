#!/bin/bash

# http://stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
MYPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIRS=`find $MYPATH -type d | grep -v \.git`

for x in $DIRS; do
  echo "Adding $x"
  export TEXINPUTS=$TEXINPUTS:$x
  export BSTINPUTS=$BSTINPUTS:$x
done
