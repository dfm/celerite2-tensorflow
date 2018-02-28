#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <Eigen/Core>

#include "celerite.h"

using namespace tensorflow;

static const char kErrMsg[] =
    "Cholesky decomposition failed";

REGISTER_OP("CeleriteFactor")
  .Attr("T: {float, double}")
  .Input("a: T")
  .Input("u: T")
  .Input("v: T")
  .Input("p: T")
  .Output("d: T")
  .Output("w: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {

    shape_inference::ShapeHandle a, u, v, p, J;

    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &u));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &v));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &p));
    TF_RETURN_IF_ERROR(c->Merge(u, v, &u));

    c->set_output(0, c->input(0));
    c->set_output(1, c->input(1));

    return Status::OK();
  });

template <typename T>
class CeleriteFactorOp : public OpKernel {
 public:
  explicit CeleriteFactorOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> c_vector_t;
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> c_matrix_t;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vector_t;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrix_t;

    const Tensor& U_t = context->input(1);
    OP_REQUIRES(context, (U_t.dims() == 2),
          errors::InvalidArgument("U should have the shape (N, J)"));
    int64 N = U_t.dim_size(0),
          J = U_t.dim_size(1);

    const Tensor& a_t = context->input(0);
    OP_REQUIRES(context, ((a_t.dims() == 1) &&
                          (a_t.dim_size(0) == N)),
        errors::InvalidArgument("a should have the shape (N)"));

    const Tensor& V_t = context->input(2);
    OP_REQUIRES(context, ((V_t.dims() == 2) &&
                          (V_t.dim_size(0) == N) &&
                          (V_t.dim_size(1) == J)),
        errors::InvalidArgument("V should have the shape (N, J)"));

    const Tensor& P_t = context->input(3);
    OP_REQUIRES(context, ((P_t.dims() == 2) &&
                          (P_t.dim_size(0) == N-1) &&
                          (P_t.dim_size(1) == J)),
        errors::InvalidArgument("P should have the shape (N-1, J)"));

    const auto U = c_matrix_t(U_t.template flat<T>().data(), N, J);
    const auto a = c_vector_t(a_t.template flat<T>().data(), N);
    const auto V = c_matrix_t(V_t.template flat<T>().data(), N, J);
    const auto P = c_matrix_t(P_t.template flat<T>().data(), N-1, J);

    // Create the outputs
    Tensor* d_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N}), &d_t));
    Tensor* W_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({N, J}), &W_t));

    auto d = vector_t(d_t->template flat<T>().data(), N);
    auto W = matrix_t(W_t->template flat<T>().data(), N, J);

    d = a;
    W = V;
    int flag = celerite::factor(U, P, d, W);
    OP_REQUIRES(context, flag == 0, errors::InvalidArgument(kErrMsg));
  }
};

#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("CeleriteFactor")                                               \
      .Device(DEVICE_CPU)                                                  \
      .TypeConstraint<type>("T"),                                          \
      CeleriteFactorOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
