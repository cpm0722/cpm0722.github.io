#!/bin/bash

PWD=`pwd`

sudo docker run -v $PWD:/root/cpm0722.github.io -p 4000:4000 -e JEKYLL_ROOTLESS=true -it jekyll/jekyll:builder /bin/bash
