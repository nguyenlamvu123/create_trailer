#site:https://tecadmin.net/install-python-3-6-ubuntu-linuxmint/
#site:https://github.com/cyrildiagne/screenpoint/issues/1
#There is a broken on Python 3.8 with opencv-python==3.4.2.17; SIFT has been removed from contrib after version 3.4.2.17 because of patent issues; so downgrade to python3.6.10
sudo apt-get install build-essential checkinstall
sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
sudo wget https://www.python.org/ftp/python/3.6.10/Python-3.6.10.tgz
sudo tar xzf Python-3.6.10.tgz#extract the downloaded package
cd Python-3.6.10
sudo ./configure --enable-optimizations
sudo make altinstall
python3.6 -V
pip install requirements.txt
