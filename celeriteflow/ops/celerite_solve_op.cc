#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <Eigen/Core>

#include "celerite.h"

using namespace tensorflow;

REGISTER_OP("CeleriteSolve")
  .Attr("T: {float, double}")
  .Input("u: T")
  .Input("p: T")
  .Input("d: T")
  .Input("w: T")
  .Input("y: T")
  .Output("z: T")
  .Output("f: T")
  .Output("g: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {

    shape_inference::DimensionHandle J, Nrhs;
    shape_inference::ShapeHandle u, p, d, w, y;

    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &u));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &p));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &d));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &w));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &y));
    TF_RETURN_IF_ERROR(c->Merge(u, w, &u));

    J = c->Dim(u, 1);
    Nrhs = c->Dim(y, 1);
    TF_RETURN_IF_ERROR(c->Multiply(J, Nrhs, &J));
    TF_RETURN_IF_ERROR(c->ReplaceDim(u, 1, J, &u));

    c->set_output(0, c->input(4));
    c->set_output(1, u);
    c->set_output(2, u);

    return Status::OK();
  });

template <typename T>
class CeleriteSolveOp : public OpKernel {
 public:
  explicit CeleriteSolveOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> c_vector_t;
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> c_matrix_t;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrix_t;

    const Tensor& U_t = context->input(0);
    const Tensor& P_t = context->input(1);
    const Tensor& d_t = context->input(2);
    const Tensor& W_t = context->input(3);
    const Tensor& Y_t = context->input(4);

    OP_REQUIRES(context, (U_t.dims() == 2),
          errors::InvalidArgument("U should have the shape (N, J)"));
    int64 N = U_t.dim_size(0), J = U_t.dim_size(1);

    OP_REQUIRES(context, ((P_t.dims() == 2) && (P_t.dim_size(0) == N-1) && (P_t.dim_size(1) == J)),
        errors::InvalidArgument("P should have the shape (N-1, J)"));

    OP_REQUIRES(context, ((d_t.dims() == 1) && (d_t.dim_size(0) == N)),
        errors::InvalidArgument("d should have the shape (N)"));

    OP_REQUIRES(context, ((W_t.dims() == 2) && (W_t.dim_size(0) == N) && (W_t.dim_size(1) == J)),
        errors::InvalidArgument("W should have the shape (N, J)"));

    OP_REQUIRES(context, ((Y_t.dims() == 2) && (Y_t.dim_size(0) == N)),
        errors::InvalidArgument("Y should have the shape (N, Nrhs)"));
    int64 Nrhs = Y_t.dim_size(1);

    const auto U = c_matrix_t(U_t.template flat<T>().data(), N, J);
    const auto P = c_matrix_t(P_t.template flat<T>().data(), N-1, J);
    const auto d = c_vector_t(d_t.template flat<T>().data(), N);
    const auto W = c_matrix_t(W_t.template flat<T>().data(), N, J);
    const auto Y = c_matrix_t(Y_t.template flat<T>().data(), N, Nrhs);

    // Create the outputs
    Tensor* Z_t = NULL;
    Tensor* F_t = NULL;
    Tensor* G_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N, Nrhs}), &Z_t));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({N, J*Nrhs}), &F_t));
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({N, J*Nrhs}), &G_t));

    auto Z = matrix_t(Z_t->template flat<T>().data(), N, Nrhs);
    auto F = matrix_t(F_t->template flat<T>().data(), N, J*Nrhs);
    auto G = matrix_t(G_t->template flat<T>().data(), N, J*Nrhs);

    Z = Y;
    celerite::solve(U, P, d, W, Z, F, G);
  }
};

#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("CeleriteSolve").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      CeleriteSolveOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
