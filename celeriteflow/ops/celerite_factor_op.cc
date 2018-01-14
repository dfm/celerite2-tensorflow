#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <Eigen/Core>

using namespace tensorflow;

static const char kErrMsg[] =
    "Cholesky decomposition was not successful. The input might not be valid.";

REGISTER_OP("CeleriteFactor")
  .Attr("T: {float, double}")
  .Input("a: T")
  .Input("u: T")
  .Input("v: T")
  .Input("p: T")
  .Output("d: T")
  .Output("w: T")
  .Output("s: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    ::tensorflow::shape_inference::ShapeHandle a, u, v, p;

    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &u));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &v));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &p));
    TF_RETURN_IF_ERROR(c->Merge(u, v, &u));

    c->set_output(0, c->input(0));
    c->set_output(1, c->input(1));
    // TODO: inform S shape (J, J)

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

    const Tensor& A_t = context->input(0);
    OP_REQUIRES(context, A_t.dims() == 1, errors::InvalidArgument("A should have the shape (N)"));
    int64 N = A_t.dim_size(0);
    const auto A = c_vector_t(A_t.template flat<T>().data(), N);

    const Tensor& U_t = context->input(1);
    OP_REQUIRES(context, ((U_t.dims() == 2) &&
                          (U_t.dim_size(0) == N)),
          errors::InvalidArgument("U should have the shape (N, J)"));
    int64 J = U_t.dim_size(1);
    const auto U = c_matrix_t(U_t.template flat<T>().data(), N, J);

    const Tensor& V_t = context->input(2);
    OP_REQUIRES(context, ((V_t.dims() == 2) &&
                          (V_t.dim_size(0) == N) &&
                          (V_t.dim_size(1) == J)),
        errors::InvalidArgument("V should have the shape (N, J)"));
    const auto V = c_matrix_t(V_t.template flat<T>().data(), N, J);

    const Tensor& P_t = context->input(3);
    OP_REQUIRES(context, ((P_t.dims() == 2) &&
                          (P_t.dim_size(0) == N-1) &&
                          (P_t.dim_size(1) == J)),
        errors::InvalidArgument("P should have the shape (N-1, J)"));
    const auto P = c_matrix_t(P_t.template flat<T>().data(), N-1, J);

    // Create the outputs
    Tensor* D_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N}), &D_t));
    auto D = vector_t(D_t->template flat<T>().data(), N);

    Tensor* W_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({N, J}), &W_t));
    auto W = matrix_t(W_t->template flat<T>().data(), N, J);

    Tensor* S_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({J, J}), &S_t));
    auto S = matrix_t(S_t->template flat<T>().data(), J, J);

    // First row
    S.setZero();
    D(0) = A(0);
    W.row(0) = V.row(0) / D(0);

    // The rest of the rows
    for (int64 n = 1; n < N; ++n) {
      // Update S = diag(P) * (S + D*W*W.T) * diag(P)
      S.noalias() += D(n-1) * W.row(n-1).transpose() * W.row(n-1);
      S.array() *= (P.row(n-1).transpose() * P.row(n-1)).array();

      // Update D = A - U * S * U.T
      W.row(n) = U.row(n) * S;
      D(n) = A(n) - W.row(n) * U.row(n).transpose();
      OP_REQUIRES(context, D(n) > T(0.0), errors::InvalidArgument(kErrMsg));

      // Update W = (V - U * S) / D
      W.row(n).noalias() -= V.row(n);
      W.row(n) /= -D(n);
    }
  }
};

#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("CeleriteFactor").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      CeleriteFactorOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
