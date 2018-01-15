#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <Eigen/Core>

using namespace tensorflow;

REGISTER_OP("CeleriteSolveGrad")
  .Attr("T: {float, double}")
  .Input("u: T")
  .Input("p: T")
  .Input("d: T")
  .Input("w: T")
  .Input("z: T")
  .Input("f: T")
  .Input("g: T")
  .Input("bz: T")
  .Input("bf: T")
  .Input("bg: T")
  .Output("bu: T")
  .Output("bp: T")
  .Output("bd: T")
  .Output("bw: T")
  .Output("by: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    ::tensorflow::shape_inference::ShapeHandle u, p, d, w;

    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &u));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &p));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &d));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &w));
    TF_RETURN_IF_ERROR(c->Merge(u, w, &u));

    c->set_output(0, c->input(0));
    c->set_output(1, c->input(1));
    c->set_output(2, c->input(2));
    c->set_output(3, c->input(3));
    c->set_output(4, c->input(4));

    return Status::OK();
  });

template <typename T>
class CeleriteSolveGradOp : public OpKernel {
 public:
  explicit CeleriteSolveGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> c_vector_t;
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> c_matrix_t;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vector_t;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrix_t;

    const Tensor& U_t = context->input(0);
    OP_REQUIRES(context, (U_t.dims() == 2),
          errors::InvalidArgument("U should have the shape (N, J)"));
    int64 N = U_t.dim_size(0), J = U_t.dim_size(1);
    const auto U = c_matrix_t(U_t.template flat<T>().data(), N, J);

    const Tensor& P_t = context->input(1);
    OP_REQUIRES(context, ((P_t.dims() == 2) &&
                          (P_t.dim_size(0) == N-1) &&
                          (P_t.dim_size(1) == J)),
        errors::InvalidArgument("P should have the shape (N-1, J)"));
    const auto P = c_matrix_t(P_t.template flat<T>().data(), N-1, J);

    const Tensor& D_t = context->input(2);
    OP_REQUIRES(context, ((D_t.dims() == 1) &&
                          (D_t.dim_size(0) == N)),
        errors::InvalidArgument("D should have the shape (N)"));
    const auto D = c_vector_t(D_t.template flat<T>().data(), N);

    const Tensor& W_t = context->input(3);
    OP_REQUIRES(context, ((W_t.dims() == 2) &&
                          (W_t.dim_size(0) == N) &&
                          (W_t.dim_size(1) == J)),
        errors::InvalidArgument("W should have the shape (N, J)"));
    const auto W = c_matrix_t(W_t.template flat<T>().data(), N, J);

    const Tensor& Y_t = context->input(4);
    OP_REQUIRES(context, (((Y_t.dims() == 1) || (Y_t.dims() == 2)) &&
                          (Y_t.dim_size(0) == N)),
        errors::InvalidArgument("Y should have the shape (N) or (N, Nrhs)"));
    bool is_oned = true;
    int64 Nrhs = 1;
    if (Y_t.dims() == 2) {
      Nrhs = Y_t.dim_size(1);
      is_oned = false;
    }
    const auto Y = c_matrix_t(Y_t.template flat<T>().data(), N, Nrhs);

    // Create the outputs
    Tensor* Z_t = NULL;
    if (is_oned)
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N}), &Z_t));
    else
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N, Nrhs}), &Z_t));
    auto Z = matrix_t(Z_t->template flat<T>().data(), N, Nrhs);

    Tensor* f_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({J, Nrhs}), &f_t));
    auto f = matrix_t(f_t->template flat<T>().data(), J, Nrhs);
    f.setZero();

    Tensor* g_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({J, Nrhs}), &g_t));
    auto g = matrix_t(g_t->template flat<T>().data(), J, Nrhs);
    g.setZero();

    Z = Y;
    for (int64 n = 1; n < N; ++n) {
      f.noalias() += W.row(n-1).transpose() * Z.row(n-1);
      f = P.row(n-1).asDiagonal() * f;
      Z.row(n).noalias() -= U.row(n) * f;
    }

    Z.array().colwise() /= D.array();

    for (int64 n = N-2; n >= 0; --n) {
      g.noalias() += U.row(n+1).transpose() * Z.row(n+1);
      g = P.row(n).asDiagonal() * g;
      Z.row(n).noalias() -= W.row(n) * g;
    }

  }
};

#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("CeleriteSolveGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      CeleriteSolveGradOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
