#!/bin/bash

#docker run --volume="$PWD:/srv/jekyll" -p 4000:4000 -i -t jekyll/jekyll jekyll serve

docker run --volume="$PWD:/srv/jekyll" -p 4000:4000 -i -t jekyll/jekyll /bin/bash
