#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <iostream>
#include <ctime>

using namespace tensorflow;

REGISTER_OP("NextHop")
.Input("possibility: float")
.Output("stay_step: int32")
.Output("next_index: int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        c->set_output(1, c->Scalar());
        return Status::OK();
});

class NextHopOp : public OpKernel {
    public:
        explicit NextHopOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensor
            const Tensor& possibility_handle = context->input(0);
            auto total_n = possibility_handle.dim_size(0);
            auto possibility = possibility_handle.flat<float>();

            int flag[total_n];
            int eff_n = 0;
            for(int i = 0; i < total_n; i++){
                if(possibility(i)>=0){
                    flag[eff_n++] = i;
                }
            }

            int stay_step = 0;
            int next_index = 0;
            std::srand(std::time(nullptr));
            for(float hop_p=0, rand_n=1;hop_p<rand_n;++stay_step){
                next_index = std::rand()%eff_n;
                hop_p = possibility(flag[next_index]);
                rand_n= ((float) std::rand()) / (float) RAND_MAX;
            }

            Tensor* res = NULL;
            TensorShape shape;
            int dims[] = {};
            TensorShapeUtils::MakeShape(dims, 0, &shape);

            OP_REQUIRES_OK(context, context->allocate_output(0, shape,
                        &res));
            auto out1 = res->flat<int32>();
            out1(0) = stay_step;

            OP_REQUIRES_OK(context, context->allocate_output(1, shape,
                        &res));
            auto out2 = res->flat<int32>();
            out2(0) = flag[next_index];
        }
};

REGISTER_KERNEL_BUILDER(Name("NextHop").Device(DEVICE_CPU), NextHopOp);
