#!usr/bin/sh

# if you want to run machine learing staff without anaconda 
# you need to install all these dependencies

sudo apt-get install python3-numpy python3-scipy python3-matplotlib ipython3 ipython3-notebook python3-pandas python3-nose  libatlas-dev libatlas3gf-base graphviz

sudo pip3  install sympy

sudo pip3  install scikit-learn

sudo pip3 install pydotplus

sudo pip3 install theano

sudo pip3 install pyprind

sudo pip3 install nltk

#change tf  version if you use not python 3.5. this is a non-gpu version
sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp34-none-linux_x86_64.whl




