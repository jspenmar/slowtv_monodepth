#!/bin/bash

YT_EXE=yt-dlp
URLS=splits/urls.txt

${YT_EXE} --no-playlist -f 136 -a ${URLS} -o "videos/%(autonumber)s.%(ext)s" --autonumber-start 0
