cd /d d:\github/mkkim85.github.io
chcp 65001
start /b bundle exec jekyll serve
timeout 5
start chrome "http://localhost:4000/"