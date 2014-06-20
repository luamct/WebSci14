
CPP    = g++
INCDIR = -I/usr/include/python2.7/ -I${PYUBLAS_INC}
LINK   = -lm -lboost_python -lboost_numpy -lpython2.7
CFLAGS = -fPIC -shared

all: _emd.so

_emd.so: _emd.cpp
	$(CPP) $(CFLAGS) -o $@ $< $(LINK) $(INCDIR)

clean:
	/bin/rm -f *.o



