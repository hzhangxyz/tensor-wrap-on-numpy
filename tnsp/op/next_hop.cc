#include <cmath>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

REGISTER_OP("NextHop")
.Input("possibility: double")
.Input("random: double")
.Output("stay_step: int32")
.Output("next_index: int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
              c->set_output(0, c->Scalar());
              c->set_output(1, c->Scalar());
              return Status::OK();
            });

std::once_flag random_seed_flag;

class NextHopOp : public OpKernel {
public:
  explicit NextHopOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& possibility_handle = context->input(0);
    auto total_n = possibility_handle.dim_size(0);
    auto possibility = possibility_handle.flat<double>();

    const Tensor& random_handle = context->input(1);
    auto random_num = random_handle.flat<double>();

    double poss_sum = 0;
    for(int i = 0; i < total_n; i++){
      if(possibility(i)<1){
        poss_sum += possibility(i);
      }else{
        poss_sum += 1;
      }
    }
    int stay_step = std::ceil(std::log(random_num(0))/std::log(1-poss_sum/total_n));

    double random = random_num(1)*poss_sum;
    double sum = 0;
    int next_index = 0;
    for(; next_index < total_n; next_index++){
      if(possibility(next_index)<1){
        sum += possibility(next_index);
      }else{
        sum += 1;
      }
      if(sum >= random){
        break;
      }
    }
    if(next_index==total_n){
      next_index = total_n - 1;
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
    out2(0) = next_index;

  }
};

REGISTER_KERNEL_BUILDER(Name("NextHop").Device(DEVICE_CPU), NextHopOp);
