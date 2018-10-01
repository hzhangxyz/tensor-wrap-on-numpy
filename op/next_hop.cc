#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <iostream>
#include <typeinfo>

using namespace tensorflow;

REGISTER_OP("NextHop")
.Input("possibility: float32")
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
            const Tensor& possibility = context->input(0);
            auto n = possibility.dim_size(0);
            auto m = possibility.dim_size(1);

            Tensor* res = NULL;
            TensorShape shape;
            int dims[] = {};
            TensorShapeUtils::MakeShape(dims, 0, &shape);

            OP_REQUIRES_OK(context, context->allocate_output(0, shape,
                        &res));
            auto out1 = res->flat<int32>();
            out1(0) = 2;

            OP_REQUIRES_OK(context, context->allocate_output(1, shape,
                        &res));
            auto out2 = res->flat<int32>();
            out2(0) = 3;
        }
};

REGISTER_KERNEL_BUILDER(Name("NextHop").Device(DEVICE_CPU), NextHopOp);
