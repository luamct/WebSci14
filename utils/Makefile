
CPP    = g++
INCDIR = -I/usr/include/python2.7/ -I${PYUBLAS_INC}
LINK   = -lm -lfftw3 -lboost_python -lboost_numpy -lpython2.7
CFLAGS = -fPIC -shared

all: _rgb2ihls.so

_rgb2ihls.so: _rgb2ihls.cpp
	$(CPP) $(CFLAGS) -o $@ $< $(LINK) $(INCDIR)

clean:
	/bin/rm -f *.o
