#!/bin/bash

git checkout source
git branch -D main
git checkout -b main
git filter-branch --subdirectory-filter _site/ -f
git push --all
git checkout source
