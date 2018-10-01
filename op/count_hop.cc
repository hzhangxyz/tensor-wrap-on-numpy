#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

REGISTER_OP("CountHop")
.Input("state: int32")
.Input("hop: int32")
.Output("number: int32")
.SetShapeFn(::tensorflow::shape_inference::ScalarShape);

class CountHopOp : public OpKernel {
    public:
        explicit CountHopOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensor
            const Tensor& spin_state = context->input(0);
            auto n = spin_state.dim_size(0);
            auto m = spin_state.dim_size(1);
            auto spin_state_data = spin_state.flat<int32>();

            const Tensor& hop_list = context->input(1);
            auto true_hop_list = hop_list.flat<int32>();
            auto hop_num = hop_list.dim_size(0);

            int sum = 0;
            Eigen::MatrixXi spin_state_tmp(n, m);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    spin_state_tmp(i, j) = spin_state_data(i*m + j);
                }
            }
            for (int i = 0; i< hop_num; i++){
                auto x= true_hop_list(2*i);
                auto y= true_hop_list(2*i + 1);
                spin_state_tmp(x, y) = 1 - spin_state_tmp(x, y);
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    if ((i!=n-1) && (spin_state_tmp.coeff(i,j) != spin_state_tmp.coeff(i+1,j))){
                        ++sum;
                    }
                    if ((j!=m-1) && (spin_state_tmp.coeff(i,j) != spin_state_tmp.coeff(i,j+1))){
                        ++sum;
                    }
                }
            }

            Tensor* res = NULL;
            TensorShape shape;
            int dims[] = {};
            TensorShapeUtils::MakeShape(dims, 0, &shape);
            OP_REQUIRES_OK(context, context->allocate_output(0, shape,
                        &res));
            auto res_flat = res->flat<int32>();
            res_flat(0) = sum;

        }
};

REGISTER_KERNEL_BUILDER(Name("CountHop").Device(DEVICE_CPU), CountHopOp);
