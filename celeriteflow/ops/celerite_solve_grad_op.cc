#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <Eigen/Core>

#include "celerite.h"

using namespace tensorflow;

REGISTER_OP("CeleriteSolveGrad")
  .Attr("T: {float, double}")
  .Input("u: T")
  .Input("p: T")
  .Input("d: T")
  .Input("w: T")
  .Input("z: T")
  .Input("bz: T")
  .Output("bu: T")
  .Output("bp: T")
  .Output("bd: T")
  .Output("bw: T")
  .Output("by: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {

    shape_inference::ShapeHandle u, p, d, w, z, bz;

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

    const Tensor& P_t = context->input(1);
    OP_REQUIRES(context, ((P_t.dims() == 2) &&
                          (P_t.dim_size(0) == N-1) &&
                          (P_t.dim_size(1) == J)),
        errors::InvalidArgument("P should have the shape (N-1, J)"));

    const Tensor& d_t = context->input(2);
    OP_REQUIRES(context, ((d_t.dims() == 1) &&
                          (d_t.dim_size(0) == N)),
        errors::InvalidArgument("d should have the shape (N)"));

    const Tensor& W_t = context->input(3);
    OP_REQUIRES(context, ((W_t.dims() == 2) &&
                          (W_t.dim_size(0) == N) &&
                          (W_t.dim_size(1) == J)),
        errors::InvalidArgument("W should have the shape (N, J)"));

    const Tensor& Z_t = context->input(4);
    OP_REQUIRES(context, (((Z_t.dims() == 1) || (Z_t.dims() == 2)) &&
                          (Z_t.dim_size(0) == N)),
        errors::InvalidArgument("Z should have the shape (N) or (N, Nrhs)"));
    int64 Nrhs = Z_t.dim_size(1);

    const Tensor& bZ_t = context->input(5);
    OP_REQUIRES(context, (((bZ_t.dims() == 1) ||
                           (bZ_t.dims() == 2 && bZ_t.dim_size(1) == Nrhs)) &&
                          (bZ_t.dim_size(0) == N)),
        errors::InvalidArgument("bZ should have the shape (N) or (N, Nrhs)"));

    const auto U = c_matrix_t(U_t.template flat<T>().data(), N, J);
    const auto P = c_matrix_t(P_t.template flat<T>().data(), N-1, J);
    const auto d = c_vector_t(d_t.template flat<T>().data(), N);
    const auto W = c_matrix_t(W_t.template flat<T>().data(), N, J);
    const auto Z = c_matrix_t(Z_t.template flat<T>().data(), N, Nrhs);
    const auto bZ = c_matrix_t(bZ_t.template flat<T>().data(), N, Nrhs);

    // Create the outputs
    Tensor* bU_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N, J}), &bU_t));
    Tensor* bP_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({N-1, J}), &bP_t));
    Tensor* bd_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({N}), &bd_t));
    Tensor* bW_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({N, J}), &bW_t));
    Tensor* bY_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, TensorShape({N, Nrhs}), &bY_t));

    auto bU = matrix_t(bU_t->template flat<T>().data(), N, J);
    auto bP = matrix_t(bP_t->template flat<T>().data(), N-1, J);
    auto bd = vector_t(bd_t->template flat<T>().data(), N);
    auto bW = matrix_t(bW_t->template flat<T>().data(), N, J);
    auto bY = matrix_t(bY_t->template flat<T>().data(), N, Nrhs);

    bU.setZero();
    bP.setZero();
    bd.setZero();
    bW.setZero();
    celerite::solve_grad(U, P, d, W, Z, bZ, bU, bP, bd, bW, bY);
  }
};

#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("CeleriteSolveGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      CeleriteSolveGradOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
