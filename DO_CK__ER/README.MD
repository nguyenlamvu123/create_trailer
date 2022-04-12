###https://github.com/wrenth04/docker-autosub
sudo docker build --tag autosub:latest .
sudo docker run -w /media/zaibachkhoa/code/create_trailer/autosub/docker-autosub/ -i -t autosub:latest
###https://github.com/kijart/docker-youtube-dl
sudo docker build -t youtube-dl .
sudo docker run --rm -v $(pwd):/media youtube-dl https://www.youtube.com/watch?v=JYwUUDdYi9I
###https://hub.docker.com/r/linuxserver/ffmpeg
sudo docker pull linuxserver/ffmpeg
sudo docker run --rm -it \
  -v $(pwd):/config \
  linuxserver/ffmpeg \
  -i /config/input.mkv \
  -c:v libx264 \
  -b:v 4M \
  -vf scale=1280:720 \
  -c:a copy \
  /config/output.mkv
