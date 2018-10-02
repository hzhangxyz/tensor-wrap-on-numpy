TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CXX=g++
FAST=-O3 -march=native
build: count_hop.so next_hop.so
clean:
	rm *.so
%.so: %.cc
	$(CXX) -std=c++11 -shared -fPIC ${TF_CFLAGS} ${TF_LFLAGS} $(FAST) -o $@ $<
test:
	python test_count_hop.py
	python test_next_hop.py
