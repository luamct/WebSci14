
# WebSci14

This project holds the code for most of components of the published work on WebSci 2014: 'The Impact of Visual Features on Online Image Diffusion'.


```
Copyright (C) 2014 by Luam Totti

This code is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation. We hope it is useful, however NO
WARRANTY IS PROVIDED. Not even the implied warranty of FITNESS FOR
FOR A PARTICULAR PURPOSE or MERCHANTABILITY.
```

# Data

The data collected and generated in this work is hosted at FigShare in the link below. The images had to be split into multiple files due to FigShare's maximum file restriction.

[The Impact of Visual Attributes on Online Image Diffusion](http://figshare.com/articles/The_Impact_of_Visual_Attributes_on_Image_Diffusion/1044282)


# Database

If you wish only to use the data provided (instead of running the extractors), this section should be enough for you. 

All the data (apart from the image files) was provided as a MySQL dump. First download and decompress it.

```
wget http://todo.link.to.db
bunzip2 websci-db.bz2
```

Now install a MySQL server and a client if you don't already have it. The following commands should create a database and restore the dump into it.

```
mysql -uuser -ppasswd -e 'CREATE DATABASE dbname'
mysql -uuser -ppasswd dbname < websci-db.sql
```

Finally, configure the `config.py` file in the project's root according to your MySQL installation. 

You should be ready to use the data directly. You can also check the `demo.py` script to see how the data can be consume in a python script. Some libraries may be required in this case.

# How to Run

The project is implemented in python with extensions written in C/C++.
The instructions provided assume a debian based Unix system. Most of the components should still work on different systems.

We first need to install some libraries for the C/C++ modules to be compiled and wrapped to python.

```
sudo apt-get install libboost-python1.49-dev libboost-python1.49.0
sudo apt-get install libfftw3-3 libfftw3-bin libfftw3-dev
```

We used the [Boost.Numpy](https://github.com/ndarray/Boost.NumPy) project for an easier python/C++ interface. The following commands should download and install it in your system. You may need to install scons.

```bash
wget https://github.com/ndarray/Boost.NumPy/archive/master.zip
unzip master.zip
cd Boost.NumPy-master/
scons
sudo scons --install-lib=/usr/lib install
```

Note the last command changes the lib path to install the *.so. This is because the default path `/usr/local/lib` may not work with python directly. Alternatively, you may leave the default configuration and include it the paths python looks for compiled modules (not tested).

Now the required python libraries must be installed. You should `pip` installed and properly configured to the python interpretable you wish to use with this project:

```
pip install https://pypi.python.org/packages/source/s/scipy/scipy-0.14.0.tar.gz
pip install https://pypi.python.org/packages/source/s/scikit-learn/scikit-learn-0.14.1.tar.gz
pip install https://pypi.python.org/packages/source/M/MySQL-python/MySQL-python-1.2.5.zip
pip install https://pypi.python.org/packages/source/m/mahotas/mahotas-1.1.0.tar.gz
pip install https://pymeanshift.googlecode.com/files/pymeanshift-0.2.1.zip
pip install https://pypi.python.org/packages/source/P/PyUblas/PyUblas-2013.1.tar.gz
```

Before compiling we need to tell the compiler where to find pyublas headers file. Run the following command to find the path:

```bash
echo `python -c 'import os,pyublas; print os.path.dirname(pyublas.__file__)'`/include
```

And append the returned value to the variable `PYUBLAS_INC` in the Makefile of the project (in the root). For example:

```
export PYUBLAS_INC = /home/luamct/websci/lib/python2.7/site-packages/pyublas/include
```

At last, you should be able to run `make` to compile de C/C++ modules.

Check the `extractor.py` file to understand the architecture of the system. Basically the script launches x working
processes (given as the only command line argument, defaulted to 1 if not provided), each processing images
marked as 'AVAILABLE' from the `jobs` table and found in the config.IMAGES_FOLDER (check `config.py`). One
example of execution is given below.

```
python extractor.py 4
```

