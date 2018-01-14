#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <Eigen/Core>

using namespace tensorflow;

REGISTER_OP("CeleriteFactorGrad")
  .Attr("T: {float, double}")
  .Input("a_real: T")
  .Input("c_real: T")
  .Input("a_comp: T")
  .Input("b_comp: T")
  .Input("c_comp: T")
  .Input("d_comp: T")
  .Input("x: T")
  .Input("diag: T")
  .Input("d_out: T")
  .Input("phi_out: T")
  .Input("u_out: T")
  .Input("w_out: T")
  .Input("s_out: T")
  .Input("d_grad: T")
  .Input("phi_grad: T")
  .Input("u_grad: T")
  .Input("w_grad: T")
  .Input("s_grad: T")
  .Output("a_real_grad: T")
  .Output("c_real_grad: T")
  .Output("a_comp_grad: T")
  .Output("b_comp_grad: T")
  .Output("c_comp_grad: T")
  .Output("d_comp_grad: T")
  .Output("x_grad: T")
  .Output("diag_grad: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    ::tensorflow::shape_inference::ShapeHandle a_real, c_real;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a_real));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &c_real));
    TF_RETURN_IF_ERROR(c->Merge(a_real, c_real, &a_real));

    ::tensorflow::shape_inference::ShapeHandle a_comp, b_comp, c_comp, d_comp;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &a_comp));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &b_comp));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &c_comp));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &d_comp));
    TF_RETURN_IF_ERROR(c->Merge(a_comp, b_comp, &a_comp));
    TF_RETURN_IF_ERROR(c->Merge(a_comp, c_comp, &a_comp));
    TF_RETURN_IF_ERROR(c->Merge(a_comp, d_comp, &a_comp));

    ::tensorflow::shape_inference::ShapeHandle x, diag;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 1, &x));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 1, &diag));
    TF_RETURN_IF_ERROR(c->Merge(x, diag, &x));

    return Status::OK();
  });

template <typename T>
class CeleriteFactorGradOp : public OpKernel {
 public:
  explicit CeleriteFactorGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    const Tensor& a_real_tensor = context->input(0);
    const Tensor& c_real_tensor = context->input(1);
    const Tensor& a_comp_tensor = context->input(2);
    const Tensor& b_comp_tensor = context->input(3);
    const Tensor& c_comp_tensor = context->input(4);
    const Tensor& d_comp_tensor = context->input(5);
    const Tensor& x_tensor = context->input(6);
    const Tensor& diag_tensor = context->input(7);

    auto a_real = a_real_tensor.template flat<T>();
    auto c_real = c_real_tensor.template flat<T>();
    auto a_comp = a_comp_tensor.template flat<T>();
    auto b_comp = b_comp_tensor.template flat<T>();
    auto c_comp = c_comp_tensor.template flat<T>();
    auto d_comp = d_comp_tensor.template flat<T>();
    auto x = x_tensor.template flat<T>();
    auto diag = diag_tensor.template flat<T>();

    const Tensor& D_tensor = context->input(8);
    const Tensor& phi_tensor = context->input(9);
    const Tensor& U_tensor = context->input(10);
    const Tensor& W_tensor = context->input(11);
    const Tensor& S_tensor = context->input(12);
    auto D = D_tensor.template flat<T>();
    auto phi = phi_tensor.template matrix<T>();
    auto U = U_tensor.template matrix<T>();
    auto W = W_tensor.template matrix<T>();
    auto S = S_tensor.template matrix<T>();

    auto D_grad   = context->input(13).template flat<T>();
    auto phi_grad = context->input(14).template matrix<T>();
    auto U_grad   = context->input(15).template matrix<T>();
    auto W_grad   = context->input(16).template matrix<T>();
    auto S_grad   = context->input(17).template matrix<T>();

    auto J_real = a_real_tensor.NumElements(),
         J_comp = a_comp_tensor.NumElements(),
         J = J_real + 2 * J_comp,
         N = x_tensor.NumElements();

    // Create the output tensors
    Tensor* a_real_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({J_real}), &a_real_grad_tensor));
    Tensor* c_real_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({J_real}), &c_real_grad_tensor));
    Tensor* a_comp_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({J_comp}), &a_comp_grad_tensor));
    Tensor* b_comp_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({J_comp}), &b_comp_grad_tensor));
    Tensor* c_comp_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, TensorShape({J_comp}), &c_comp_grad_tensor));
    Tensor* d_comp_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, TensorShape({J_comp}), &d_comp_grad_tensor));
    Tensor* x_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, TensorShape({N}), &x_grad_tensor));
    Tensor* diag_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(7, TensorShape({N}), &diag_grad_tensor));

    auto a_real_grad = a_real_grad_tensor->template flat<T>();
    auto c_real_grad = c_real_grad_tensor->template flat<T>();
    auto a_comp_grad = a_comp_grad_tensor->template flat<T>();
    auto b_comp_grad = b_comp_grad_tensor->template flat<T>();
    auto c_comp_grad = c_comp_grad_tensor->template flat<T>();
    auto d_comp_grad = d_comp_grad_tensor->template flat<T>();
    auto x_grad      = x_grad_tensor->     template flat<T>();
    auto diag_grad   = diag_grad_tensor->  template flat<T>();

  //  T a_sum = T(0);
  //  for (int64 j = 0; j < J_real; ++j) a_sum += a_real(j);
  //  for (int64 j = 0; j < J_comp; ++j) a_sum += a_comp(j);

  //  // Set the diagonal
  //  for (int64 n = 0; n < N; ++n) D(n) = diag(n) + a_sum;
  //  auto Dn = D(0);

  //  // Special case for jitter only.
  //  if (J == 0) return;

  //  // Compute the values at x[0]
  //  {
  //    T value = 1.0 / Dn,
  //      t = x(0);
  //    for (int j = 0; j < J_real; ++j) {
  //      W(j, 0) = value;
  //    }
  //    for (int j = 0, k = J_real; j < J_comp; ++j, k += 2) {
  //      T d = d_comp(j) * t;
  //      W(k,   0) = cos(d)*value;
  //      W(k+1, 0) = sin(d)*value;
  //    }
  //  }

  //  // Start the main loop
  //  for (int64 n = 1; n < N; ++n) {
  //    T t = x(n),
  //      dx = t - x(n-1);
  //    for (int64 j = 0; j < J_real; ++j) {
  //      phi(j, n-1) = exp(-c_real(j)*dx);
  //      U(j, n-1) = a_real(j);
  //      W(j, n) = T(1.0);
  //    }
  //    for (int64 j = 0, k = J_real; j < J_comp; ++j, k += 2) {
  //      T a = a_comp(j),
  //        b = b_comp(j),
  //        d = d_comp(j) * t,
  //        cd = cos(d),
  //        sd = sin(d);
  //      T value = exp(-c_comp(j)*dx);
  //      phi(k,   n-1) = value;
  //      phi(k+1, n-1) = value;
  //      U(k,   n-1) = a*cd + b*sd;
  //      U(k+1, n-1) = a*sd - b*cd;
  //      W(k,   n) = cd;
  //      W(k+1, n) = sd;
  //    }

  //    for (int64 j = 0; j < J; ++j) {
  //      T phij = phi(j, n-1),
  //        xj = Dn*W(j, n-1);
  //      for (int64 k = 0; k <= j; ++k) {
  //        S(k, j) = phij*(phi(k, n-1)*(S(k, j) + xj*W(k, n-1)));
  //      }
  //    }

  //    Dn = D(n);
  //    for (int64 j = 0; j < J; ++j) {
  //      T uj = U(j, n-1),
  //        xj = W(j, n);
  //      for (int64 k = 0; k < j; ++k) {
  //        T tmp = U(k, n-1) * S(k, j);
  //        Dn -= 2.0*(uj*tmp);
  //        xj -= tmp;
  //        W(k, n) -= uj*S(k, j);
  //      }
  //      T tmp = uj*S(j, j);
  //      Dn -= uj*tmp;
  //      W(j, n) = xj - tmp;
  //    }
  //    OP_REQUIRES(context, Dn > T(0.0), errors::InvalidArgument(kErrMsg));
  //    D(n) = Dn;
  //    for (int64 j = 0; j < J; ++j) W(j, n) /= Dn;
  //  }

    T Dn, dDn = T(0.0);
    Eigen::Matrix<T, Eigen::Dynamic, 1> Wn, dWn(J);
    for (int64 n = N-1; n >= 1; --n) {

      // for (int64 j = 0; j < J; ++j) W(j, n) /= Dn;
      dDn = T(0.0);
      dWn.setZero();
      Dn = D(n);
      for (int64 j = J-1; j >= 0; --j) {
        dDn += -W_grad(j, n) * W(j, n) / (Dn * Dn);
        dWn(j) += W_grad(j, n) / Dn;
        Wn(j) = W(j, n) * Dn;
      }



    }
  }
};

#define REGISTER_KERNEL(type)                                                  \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("CeleriteFactorGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      CeleriteFactorGradOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
