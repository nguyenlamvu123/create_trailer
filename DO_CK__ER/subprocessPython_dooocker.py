import subprocess as s 

class Doc_ker:
    def __init__(self):
        self.sudo_password = '12'
        self.cmd1 = s.Popen(
            ['echo', self.sudo_password],
            stdout=s.PIPE
            )

    def thuchien(
        self,
        commmand = 'docker images -a',
        ):
        commmand = commmand.split()

        cmd2 = s.Popen(
            ['sudo','-S'] + commmand,
            stdin=self.cmd1.stdout,
            stdout=s.PIPE
            )

        return cmd2.stdout.read().decode()
    
    def danhsach_images(self):
        return self.thuchien()
    
    def danhsach_containers_trongmay(self):
        return self.thuchien(
            commmand = 'docker ps -a',
            )
    
    def xoatoanbo_images(self):#failed 
        return self.thuchien(
            commmand = 'docker rmi $(sudo docker images -a -q)',
            )
    
    def xoatoanbo_containers_trongmay(self):#failed 
        return self.thuchien(
            commmand = 'docker rm $(sudo docker ps -a -q)',
            )
    
    def download_youtube(self):#failed
"""https://github.com/kijart/docker-youtube-dl
sudo docker build -t youtube-dl .
sudo docker run --rm -v $(pwd):/media youtube-dl https://www.youtube.com/watch?v=JYwUUDdYi9I
###sudo docker pull kijart/youtube-dl
###sudo docker run --rm -v $(pwd):/media kijart/youtube-dl https://www.youtube.com/watch\?v\=JYwUUDdYi9I"""
        self.thuchien(
            commmand = 'docker pull kijart/youtube-dl',
            )        
        return self.thuchien(
                    commmand = 'docker run --rm -v '+ \
                    '$(pwd):/media kijart/youtube-dl '+ \
                    'https://www.youtube.com/watch?v=RjmXFtz6p4U',
                    )
        
dockeeer = Doc_ker()

##for n in dockeeer.danhsach_images().strip().split('\n'):
##    print(n)
##for n in dockeeer.danhsach_containers_trongmay().strip().split('\n'):
##    print(n)
##for n in dockeeer.xoatoanbo_images().strip().split('\n'):
##    print(n)
##for n in dockeeer.xoatoanbo_containers_trongmay().strip().split('\n'):
##    print(n)
for n in dockeeer.download_youtube().strip().split('\n'):
    print(n)

#####https://github.com/wrenth04/docker-autosub
##sudo docker build --tag autosub:latest .
##sudo docker run --rm -w /media/zaibachkhoa/code/create_trailer/autosub/docker-autosub/ -i -t autosub:latest
###sudo docker pull wrenth04/autosub
#####https://github.com/kijart/docker-youtube-dl
##sudo docker build -t youtube-dl .
##sudo docker run --rm -v $(pwd):/media youtube-dl https://www.youtube.com/watch?v=JYwUUDdYi9I
###sudo docker pull kijart/youtube-dl
###sudo docker run --rm -v $(pwd):/media kijart/youtube-dl https://www.youtube.com/watch\?v\=JYwUUDdYi9I
#####https://hub.docker.com/r/linuxserver/ffmpeg
##sudo docker pull linuxserver/ffmpeg
##sudo docker run --rm -it \
##  -v $(pwd):/config \
##  linuxserver/ffmpeg \
##  -i /config/input.mkv \
##  -c:v libx264 \
##  -b:v 4M \
##  -vf scale=1280:720 \
##  -c:a copy \
##  /config/output.mkv
###sudo docker run --rm -it \
###  -v "$(pwd)":/config \
###  linuxserver/ffmpeg \
###  -i /config/uavdemo_sta.avi /config/video1.mp4
#####https://github.com/ricktorzynski/ocr-tesseract-docker
#####https://stackoverflow.com/questions/63048908/how-do-i-install-a-new-language-pack-for-tesseract-on-windows
#####https://stackoverflow.com/questions/14800730/tesseract-running-error
##sudo docker build -t ocr-tesseract-docker .
##sudo docker run -d -p 5000:5000 ocr-tesseract-docker
###sudo docker pull ricktorzynski/ocr-tesseract-docker
###sudo docker run -d -p 5000:5000 ricktorzynski/ocr-tesseract-docker
