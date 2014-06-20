
export PYUBLAS_INC = /home/luamct/anaconda/lib/python2.7/site-packages/pyublas/include
SUBDIRS = sharpness utils color
     
.PHONY: subdirs $(SUBDIRS)
     
subdirs: $(SUBDIRS)
 
$(SUBDIRS):
	$(MAKE) -C $@

