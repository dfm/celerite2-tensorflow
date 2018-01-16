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

    ::tensorflow::shape_inference::ShapeHandle u, p, d, w, z, bz;

    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &u));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &p));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &d));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &w));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &z));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 2, &bz));
    TF_RETURN_IF_ERROR(c->Merge(u, w, &u));
    TF_RETURN_IF_ERROR(c->Merge(z, bz, &bz));

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

    const Tensor& Z_t = context->input(4);
    OP_REQUIRES(context, (((Z_t.dims() == 1) || (Z_t.dims() == 2)) &&
                          (Z_t.dim_size(0) == N)),
        errors::InvalidArgument("Z should have the shape (N) or (N, Nrhs)"));
    int64 Nrhs = Z_t.dim_size(1);
    const auto Z = c_matrix_t(Z_t.template flat<T>().data(), N, Nrhs);

    const Tensor& f_t = context->input(5);
    OP_REQUIRES(context, ((f_t.dims() == 2) &&
                          (f_t.dim_size(0) == J) &&
                          (f_t.dim_size(1) == Nrhs)),
        errors::InvalidArgument("f should have the shape (J, Nrhs)"));
    const auto f = c_matrix_t(f_t.template flat<T>().data(), J, Nrhs);

    const Tensor& g_t = context->input(6);
    OP_REQUIRES(context, ((g_t.dims() == 2) &&
                          (g_t.dim_size(0) == J) &&
                          (g_t.dim_size(1) == Nrhs)),
        errors::InvalidArgument("g should have the shape (J, Nrhs)"));
    const auto g = c_matrix_t(g_t.template flat<T>().data(), J, Nrhs);

    const Tensor& bZ_t = context->input(7);
    OP_REQUIRES(context, (((bZ_t.dims() == 1) ||
                           (bZ_t.dims() == 2 && bZ_t.dim_size(1) == Nrhs)) &&
                          (bZ_t.dim_size(0) == N)),
        errors::InvalidArgument("bZ should have the shape (N) or (N, Nrhs)"));
    const auto bZ = c_matrix_t(bZ_t.template flat<T>().data(), N, Nrhs);

    const Tensor& bf_t = context->input(8);
    OP_REQUIRES(context, ((bf_t.dims() == 2) &&
                          (bf_t.dim_size(0) == J) &&
                          (bf_t.dim_size(1) == Nrhs)),
        errors::InvalidArgument("bf should have the shape (J, Nrhs)"));
    const auto bf = c_matrix_t(bf_t.template flat<T>().data(), J, Nrhs);

    const Tensor& bg_t = context->input(9);
    OP_REQUIRES(context, ((bg_t.dims() == 2) &&
                          (bg_t.dim_size(0) == J) &&
                          (bg_t.dim_size(1) == Nrhs)),
        errors::InvalidArgument("bg should have the shape (J, Nrhs)"));
    const auto bg = c_matrix_t(bg_t.template flat<T>().data(), J, Nrhs);

    // Create the outputs
    Tensor* bU_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N, J}), &bU_t));
    auto bU = matrix_t(bU_t->template flat<T>().data(), N, J);
    //bU.setZero();

    Tensor* bP_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({N-1, J}), &bP_t));
    auto bP = matrix_t(bP_t->template flat<T>().data(), N-1, J);
    //bP.setZero();

    Tensor* bD_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({N}), &bD_t));
    auto bD = vector_t(bD_t->template flat<T>().data(), N);

    Tensor* bW_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({N, J}), &bW_t));
    auto bW = matrix_t(bW_t->template flat<T>().data(), N, J);
    //bW.setZero();

    Tensor* bY_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, TensorShape({N, Nrhs}), &bY_t));
    auto bY = matrix_t(bY_t->template flat<T>().data(), N, Nrhs);

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      bf_ = bf, f_ = f, bg_ = bg, g_ = g, Z_ = Z;

    bY = bZ;
    bU.row(0).setZero();

    for (int64 n = 0; n <= N-2; ++n) {
      // Grad of: Z.row(n).noalias() -= W.row(n) * g;
      bW.row(n).noalias() = -bY.row(n) * g_.transpose();
      bg_ -= W.row(n).transpose() * bY.row(n);

      // Inverse of: Z.row(n).noalias() -= W.row(n) * g;
      Z_.row(n).noalias() += W.row(n) * g_;

      // Inverse of: g = P.row(n).asDiagonal() * g;
      g_ = P.row(n).asDiagonal().inverse() * g_;

      // Grad of: g = P.row(n).asDiagonal() * g;
      bP.row(n).noalias() = (bg_ * g_.transpose()).diagonal();
      bg_ = P.row(n).asDiagonal() * bg_;

      // Inverse of: g.noalias() += U.row(n+1).transpose() * Z.row(n+1);
      g_.noalias() -= U.row(n+1).transpose() * Z_.row(n+1);

      // Grad of: g.noalias() += U.row(n+1).transpose() * Z.row(n+1);
      bU.row(n+1).noalias() = Z_.row(n+1) * bg_.transpose();
      bY.row(n+1).noalias() += U.row(n+1) * bg_;
    }

    bW.row(N-1).setZero();

    bY.array().colwise() /= D.array();
    bD = -(Z_.array() * bY.array()).rowwise().sum();

    // Inverse of: Z.array().colwise() /= D.array();
    Z_.array().colwise() *= D.array();

    for (int64 n = N-1; n >= 1; --n) {
      // Grad of: Z.row(n).noalias() -= U.row(n) * f;
      bU.row(n).noalias() -= bY.row(n) * f_.transpose();
      bf_ -= U.row(n).transpose() * bY.row(n);

      // Inverse of: f = P.row(n-1).asDiagonal() * f;
      f_ = P.row(n-1).asDiagonal().inverse() * f_;

      // Grad of: f = P.row(n-1).asDiagonal() * f;
      bP.row(n-1).noalias() += (bf_ * f_.transpose()).diagonal();
      bf_ = P.row(n-1).asDiagonal() * bf_;

      // Inverse of: f.noalias() += W.row(n-1).transpose() * Z.row(n-1);
      f_.noalias() -= W.row(n-1).transpose() * Z_.row(n-1);

      // Grad of: f.noalias() += W.row(n-1).transpose() * Z.row(n-1);
      bW.row(n-1).noalias() += Z_.row(n-1) * bf_.transpose();
      bY.row(n-1).noalias() += W.row(n-1) * bf_;
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
