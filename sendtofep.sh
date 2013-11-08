#!/bin/bash

make clean && rsync -avzh --stats --progress . mihai.barbulescu@fep.grid.pub.ro:~/liquid_dsp
echo "All done"
