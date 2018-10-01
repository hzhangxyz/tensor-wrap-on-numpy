TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
build:
	g++ -std=c++11 -shared count_hop.cc -o count_hop.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -O2 -march=native
test:
	python test.py
