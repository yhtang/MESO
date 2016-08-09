#!/bin/bash

cat $1 | grep $2 | awk -F' ' '{ sum += $4 } END { if (NR > 0) print sum / NR }'
