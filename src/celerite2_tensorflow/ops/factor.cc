#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <Eigen/Core>

#include <celerite2/celerite2.h>

using namespace tensorflow;

static const char kErrMsg[] = "Cholesky decomposition failed";

REGISTER_OP("CeleriteFactor")
   .Input("a: double")
   .Input("u: double")
   .Input("v: double")
   .Input("p: double")
   .Output("d: double")
   .Output("w: double")
   .Output("s: double")
   .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
     ::tensorflow::shape_inference::DimensionHandle N, J;
     ::tensorflow::shape_inference::ShapeHandle a, U, V, P;

     TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a));
     TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &U));
     TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &V));
     TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &P));
     TF_RETURN_IF_ERROR(c->Merge(U, V, &U));

     N = c->Dim(U, 0);
     J = c->Dim(U, 1);

     c->set_output(0, c->input(0));
     c->set_output(1, c->input(1));
     c->set_output(2, c->MakeShape({N, J, J}));

     return Status::OK();
   });

REGISTER_OP("CeleriteFactorRev")
   .Input("a: double")
   .Input("u: double")
   .Input("v: double")
   .Input("p: double")
   .Input("d: double")
   .Input("w: double")
   .Input("s: double")
   .Input("bd: double")
   .Input("bw: double")
   .Output("ba: double")
   .Output("bu: double")
   .Output("bv: double")
   .Output("bp: double")
   .SetShapeFn([](shape_inference::InferenceContext *c) {
     ::tensorflow::shape_inference::ShapeHandle a, U, V, P, d, W, S, bd, bW;

     TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a));
     TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &U));
     TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &V));
     TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &P));
     TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &d));
     TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 2, &W));
     TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 3, &S));
     TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 1, &bd));
     TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 2, &bW));

     c->set_output(0, a);
     c->set_output(1, U);
     c->set_output(2, V);
     c->set_output(3, P);

     return Status::OK();
   });

template <typename T>
class CeleriteFactorOp : public OpKernel {
  public:
  explicit CeleriteFactorOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> c_vector_t;
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> c_matrix_t;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vector_t;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrix_t;

    const Tensor &a_t = context->input(0);
    const Tensor &U_t = context->input(1);
    const Tensor &V_t = context->input(2);
    const Tensor &P_t = context->input(3);

    OP_REQUIRES(context, (U_t.dims() == 2), errors::InvalidArgument("U should have the shape (N, J)"));
    int64 N = U_t.dim_size(0), J = U_t.dim_size(1);

    OP_REQUIRES(context, ((a_t.dims() == 1) && (a_t.dim_size(0) == N)), errors::InvalidArgument("a should have the shape (N)"));
    OP_REQUIRES(context, ((V_t.dims() == 2) && (V_t.dim_size(0) == N) && (V_t.dim_size(1) == J)),
                errors::InvalidArgument("V should have the shape (N, J)"));
    OP_REQUIRES(context, ((P_t.dims() == 2) && (P_t.dim_size(0) == N - 1) && (P_t.dim_size(1) == J)),
                errors::InvalidArgument("P should have the shape (N-1, J)"));

    const auto U = c_matrix_t(U_t.template flat<T>().data(), N, J);
    const auto a = c_vector_t(a_t.template flat<T>().data(), N);
    const auto V = c_matrix_t(V_t.template flat<T>().data(), N, J);
    const auto P = c_matrix_t(P_t.template flat<T>().data(), N - 1, J);

    // Create the outputs
    Tensor *d_t = NULL;
    Tensor *W_t = NULL;
    Tensor *S_t = NULL;

    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N}), &d_t));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({N, J}), &W_t));
    OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({N, J, J}), &S_t));

    auto d = vector_t(d_t->template flat<T>().data(), N);
    auto W = matrix_t(W_t->template flat<T>().data(), N, J);
    auto S = matrix_t(S_t->template flat<T>().data(), N, J * J);

    int flag = celerite2::core::factor(a, U, V, P, d, W, S);
    OP_REQUIRES(context, flag == 0, errors::InvalidArgument(kErrMsg));
  }
};

template <typename T>
class CeleriteFactorRevOp : public OpKernel {
  public:
  explicit CeleriteFactorRevOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> c_vector_t;
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> c_matrix_t;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vector_t;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrix_t;

    const Tensor &a_t  = context->input(0);
    const Tensor &U_t  = context->input(1);
    const Tensor &V_t  = context->input(2);
    const Tensor &P_t  = context->input(3);
    const Tensor &d_t  = context->input(4);
    const Tensor &W_t  = context->input(5);
    const Tensor &S_t  = context->input(6);
    const Tensor &bd_t = context->input(7);
    const Tensor &bW_t = context->input(8);

    OP_REQUIRES(context, U_t.dims() == 2, errors::InvalidArgument("U must be a matrix"));
    int64 N = U_t.dim_size(0), J = U_t.dim_size(1);

    OP_REQUIRES(context, ((a_t.dims() == 1) && (a_t.dim_size(0) == N)), errors::InvalidArgument("a should have shape (N)"));
    OP_REQUIRES(context, ((V_t.dims() == 2) && (V_t.dim_size(0) == N) && (V_t.dim_size(1) == J)),
                errors::InvalidArgument("V should have shape (N, J)"));

    OP_REQUIRES(context, ((P_t.dims() == 2) && (P_t.dim_size(0) == N - 1) && (P_t.dim_size(1) == J)),
                errors::InvalidArgument("P should have shape (N-1, J)"));

    OP_REQUIRES(context, ((d_t.dims() == 1) && (d_t.dim_size(0) == N)), errors::InvalidArgument("d should have shape (N)"));

    OP_REQUIRES(context, ((W_t.dims() == 2) && (W_t.dim_size(0) == N) && (W_t.dim_size(1) == J)),
                errors::InvalidArgument("W should have shape (N, J)"));

    OP_REQUIRES(context, ((S_t.dims() == 3) && (S_t.dim_size(0) == N) && (S_t.dim_size(1) == J) && (S_t.dim_size(2) == J)),
                errors::InvalidArgument("S should have shape (N, J, J)"));

    OP_REQUIRES(context, ((bd_t.dims() == 1) && (bd_t.dim_size(0) == N)), errors::InvalidArgument("bd should have shape (N)"));

    OP_REQUIRES(context, ((bW_t.dims() == 2) && (bW_t.dim_size(0) == N) && (bW_t.dim_size(1) == J)),
                errors::InvalidArgument("bW should have shape (N, J)"));

    const auto a  = c_vector_t(a_t.template flat<T>().data(), N);
    const auto U  = c_matrix_t(U_t.template flat<T>().data(), N, J);
    const auto V  = c_matrix_t(V_t.template flat<T>().data(), N, J);
    const auto P  = c_matrix_t(P_t.template flat<T>().data(), N - 1, J);
    const auto d  = c_vector_t(d_t.template flat<T>().data(), N);
    const auto W  = c_matrix_t(W_t.template flat<T>().data(), N, J);
    const auto S  = c_matrix_t(S_t.template flat<T>().data(), N, J * J);
    const auto bd = c_vector_t(bd_t.template flat<T>().data(), N);
    const auto bW = c_matrix_t(bW_t.template flat<T>().data(), N, J);

    // Create the outputs
    Tensor *ba_t = NULL;
    Tensor *bU_t = NULL;
    Tensor *bV_t = NULL;
    Tensor *bP_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N}), &ba_t));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({N, J}), &bU_t));
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({N, J}), &bV_t));
    OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({N - 1, J}), &bP_t));

    auto ba = vector_t(ba_t->template flat<T>().data(), N);
    auto bU = matrix_t(bU_t->template flat<T>().data(), N, J);
    auto bV = matrix_t(bV_t->template flat<T>().data(), N, J);
    auto bP = matrix_t(bP_t->template flat<T>().data(), N - 1, J);

    celerite2::core::factor_rev(a, U, V, P, d, W, S, bd, bW, ba, bU, bV, bP);
  }
};

REGISTER_KERNEL_BUILDER(Name("CeleriteFactor").Device(DEVICE_CPU), CeleriteFactorOp<double>);
REGISTER_KERNEL_BUILDER(Name("CeleriteFactorRev").Device(DEVICE_CPU), CeleriteFactorRevOp<double>);

#undef REGISTER_KERNEL
