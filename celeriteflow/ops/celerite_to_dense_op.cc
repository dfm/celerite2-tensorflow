#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_types.h"

#include <Eigen/Core>

#include "celerite.h"

using namespace tensorflow;

REGISTER_OP("CeleriteToDense")
  .Attr("T: {float, double}")
  .Input("a: T")
  .Input("u: T")
  .Input("v: T")
  .Input("p: T")
  .Output("k: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {

    shape_inference::DimensionHandle N;
    shape_inference::ShapeHandle a, u, v, p;

    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &u));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &v));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &p));
    TF_RETURN_IF_ERROR(c->Merge(u, v, &u));

    // Compute the shape for K
    N = c->Dim(u, 0);
    TF_RETURN_IF_ERROR(c->ReplaceDim(u, 1, N, &u));
    c->set_output(0, u);

    return Status::OK();
  });

template <typename T>
class CeleriteToDenseOp : public OpKernel {
 public:
  explicit CeleriteToDenseOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> c_vector_t;
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> c_matrix_t;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrix_t;

    const Tensor& a_t = context->input(0);
    const Tensor& U_t = context->input(1);
    const Tensor& V_t = context->input(2);
    const Tensor& P_t = context->input(3);

    OP_REQUIRES(context, (U_t.dims() == 2),
          errors::InvalidArgument("U should have the shape (N, J)"));

    int64 N = U_t.dim_size(0),
          J = U_t.dim_size(1);

    OP_REQUIRES(context, ((a_t.dims() == 1) && (a_t.dim_size(0) == N)),
        errors::InvalidArgument("a should have the shape (N)"));
    OP_REQUIRES(context, ((V_t.dims() == 2) && (V_t.dim_size(0) == N) && (V_t.dim_size(1) == J)),
        errors::InvalidArgument("V should have the shape (N, J)"));
    OP_REQUIRES(context, ((P_t.dims() == 2) && (P_t.dim_size(0) == N-1) && (P_t.dim_size(1) == J)),
        errors::InvalidArgument("P should have the shape (N-1, J)"));

    const auto U = c_matrix_t(U_t.template flat<T>().data(), N, J);
    const auto a = c_vector_t(a_t.template flat<T>().data(), N);
    const auto V = c_matrix_t(V_t.template flat<T>().data(), N, J);
    const auto P = c_matrix_t(P_t.template flat<T>().data(), N-1, J);

    // Create the outputs
    Tensor* K_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N, N}), &K_t));

    auto K = matrix_t(K_t->template flat<T>().data(), N, N);

    celerite::to_dense(a, U, V, P, K);
  }
};

#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("CeleriteToDense")                                              \
      .Device(DEVICE_CPU)                                                  \
      .TypeConstraint<type>("T"),                                          \
      CeleriteToDenseOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
