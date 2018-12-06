TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CHECK_ICC=$(shell which icc 1>/dev/null 2>&1; echo $$?)
ifeq ($(CHECK_ICC),0)
	CXX=icc
	FAST=-fast
else
	CXX=g++
	FAST=-O3 -march=native
endif
build: count_hop.so next_hop.so
clean:
	rm *.so
%.so: %.cc
	$(CXX) -std=c++11 -shared -fPIC ${TF_CFLAGS} ${TF_LFLAGS} $(FAST) -o $@ $<
test:
	python test_count_hop.py
	python test_next_hop.py
test_com:
	echo ${CXX}
