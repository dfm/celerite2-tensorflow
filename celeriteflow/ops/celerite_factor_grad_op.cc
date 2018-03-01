#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <Eigen/Core>

#include "celerite.h"

using namespace tensorflow;

REGISTER_OP("CeleriteFactorGrad")
  .Attr("T: {float, double}")
  .Input("u: T")
  .Input("p: T")
  .Input("d: T")
  .Input("w: T")
  .Input("s: T")
  .Input("bd: T")
  .Input("bw: T")
  .Output("ba: T")
  .Output("bu: T")
  .Output("bv: T")
  .Output("bp: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {

    shape_inference::ShapeHandle u, p, d, w, s, bd, bw;

    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &u));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &p));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &d));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &w));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &s));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &bd));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 2, &bw));

    TF_RETURN_IF_ERROR(c->Merge(u, w, &u));
    TF_RETURN_IF_ERROR(c->Merge(u, bw, &u));
    TF_RETURN_IF_ERROR(c->Merge(d, bd, &d));

    c->set_output(0, c->input(2));
    c->set_output(1, c->input(0));
    c->set_output(2, c->input(0));
    c->set_output(3, c->input(1));

    return Status::OK();
  });

template <typename T>
class CeleriteFactorGradOp : public OpKernel {
 public:
  explicit CeleriteFactorGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> c_vector_t;
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> c_matrix_t;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vector_t;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrix_t;

    const Tensor& U_t = context->input(0);
    const Tensor& P_t = context->input(1);
    const Tensor& d_t = context->input(2);
    const Tensor& W_t = context->input(3);
    const Tensor& S_t = context->input(4);
    const Tensor& bd_t = context->input(5);
    const Tensor& bW_t = context->input(6);

    OP_REQUIRES(context, U_t.dims() == 2, errors::InvalidArgument("U should be a matrix"));
    int64 N = U_t.dim_size(0),
          J = U_t.dim_size(1);

    OP_REQUIRES(context, ((P_t.dims() == 2) &&
                          (P_t.dim_size(0) == N-1) &&
                          (P_t.dim_size(1) == J)),
          errors::InvalidArgument("P should have shape (N-1, J)"));

    OP_REQUIRES(context, ((d_t.dims() == 1) && (d_t.dim_size(0) == N)),
        errors::InvalidArgument("d should have shape (N)"));

    OP_REQUIRES(context, ((W_t.dims() == 2) &&
                          (W_t.dim_size(0) == N) &&
                          (W_t.dim_size(1) == J)),
          errors::InvalidArgument("W should have shape (N, J)"));

    OP_REQUIRES(context, ((S_t.dims() == 2) &&
                          (S_t.dim_size(0) == N) &&
                          (S_t.dim_size(1) == J*J)),
          errors::InvalidArgument("S should have shape (N, J*J)"));

    OP_REQUIRES(context, ((bd_t.dims() == 1) && (bd_t.dim_size(0) == N)),
        errors::InvalidArgument("bd should have shape (N)"));

    OP_REQUIRES(context, ((bW_t.dims() == 2) &&
                          (bW_t.dim_size(0) == N) &&
                          (bW_t.dim_size(1) == J)),
          errors::InvalidArgument("bW should have shape (N, J)"));

    const auto U = c_matrix_t(U_t.template flat<T>().data(), N, J);
    const auto P = c_matrix_t(P_t.template flat<T>().data(), N-1, J);
    const auto d = c_vector_t(d_t.template flat<T>().data(), N);
    const auto W = c_matrix_t(W_t.template flat<T>().data(), N, J);
    const auto S = c_matrix_t(S_t.template flat<T>().data(), N, J*J);
    const auto bd = c_vector_t(bd_t.template flat<T>().data(), N);
    const auto bW = c_matrix_t(bW_t.template flat<T>().data(), N, J);

    // Create the outputs
    Tensor* ba_t = NULL;
    Tensor* bU_t = NULL;
    Tensor* bV_t = NULL;
    Tensor* bP_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N}), &ba_t));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({N, J}), &bU_t));
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({N, J}), &bV_t));
    OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({N-1, J}), &bP_t));

    auto ba = vector_t(ba_t->template flat<T>().data(), N);
    auto bU = matrix_t(bU_t->template flat<T>().data(), N, J);
    auto bV = matrix_t(bV_t->template flat<T>().data(), N, J);
    auto bP = matrix_t(bP_t->template flat<T>().data(), N-1, J);

    bU.setZero();
    bP.setZero();
    ba = bd;
    bV = bW;
    celerite::factor_grad(U, P, d, W, S, bU, bP, ba, bV);
  }
};

#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("CeleriteFactorGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      CeleriteFactorGradOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
