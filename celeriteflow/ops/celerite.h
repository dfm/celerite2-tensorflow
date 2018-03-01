#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace celerite {

template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
int factor (
  const Eigen::MatrixBase<T1>& U,  // (N, J)
  const Eigen::MatrixBase<T2>& P,  // (N-1, J)
  Eigen::MatrixBase<T3>& d,        // (N);    initially set to A
  Eigen::MatrixBase<T4>& W,        // (N, J); initially set to V
  Eigen::MatrixBase<T5>& S         // (N, J*J)
) {
  int N = U.rows(), J = U.cols();

  Eigen::Matrix<typename T1::Scalar, 1, T1::ColsAtCompileTime> tmp(1, J);
  Eigen::Matrix<typename T1::Scalar, T1::ColsAtCompileTime, T1::ColsAtCompileTime> S_(J, J);

  // First row
  S_.setZero();
  S.row(0).setZero();
  W.row(0) /= d(0);

  // The rest of the rows
  for (int n = 1; n < N; ++n) {
    // Update S = diag(P) * (S + d*W*W.T) * diag(P)
    S_.noalias() += d(n-1) * W.row(n-1).transpose() * W.row(n-1);
    S_.array() *= (P.row(n-1).transpose() * P.row(n-1)).array();
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < J; ++k)
        S(n, j*J+k) = S_(j, k);

    // Update d = a - U * S * U.T
    tmp = U.row(n) * S_;
    d(n) -= tmp * U.row(n).transpose();
    if (d(n) <= 0.0) return n;

    // Update W = (V - U * S) / d
    W.row(n).noalias() -= tmp;
    W.row(n) /= d(n);
  }

  return 0;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
void factor_grad (
  const Eigen::MatrixBase<T1>& U,   // (N, J)
  const Eigen::MatrixBase<T2>& P,   // (N-1, J)
  const Eigen::MatrixBase<T3>& d,   // (N)
  const Eigen::MatrixBase<T1>& W,   // (N, J)
  const Eigen::MatrixBase<T4>& S,   // (N, J*J)

  Eigen::MatrixBase<T5>& bU,        // (N, J)
  Eigen::MatrixBase<T6>& bP,        // (N-1, J)
  Eigen::MatrixBase<T7>& ba,        // (N)
  Eigen::MatrixBase<T5>& bV         // (N, J)
) {
  int N = U.rows(), J = U.cols();

  // Make local copies of the gradients that we need.
  typedef Eigen::Matrix<typename T1::Scalar, T1::ColsAtCompileTime, T1::ColsAtCompileTime, T1::IsRowMajor> S_t;
  S_t S_(J, J), bS = S_t::Zero(J, J);
  Eigen::Matrix<typename T1::Scalar, T1::ColsAtCompileTime, 1> bSWT;

  bV.array().colwise() /= d.array();
  for (int n = N-1; n > 0; --n) {
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < J; ++k)
        S_(j, k) = S(n, j*J+k);

    // Step 6
    ba(n) -= W.row(n) * bV.row(n).transpose();
    bU.row(n).noalias() = -(bV.row(n) + 2.0 * ba(n) * U.row(n)) * S_;
    bS.noalias() -= U.row(n).transpose() * (bV.row(n) + ba(n) * U.row(n));

    // Step 4
    S_ *= P.row(n-1).asDiagonal().inverse();
    bP.row(n-1).noalias() = (bS * S_ + S_.transpose() * bS).diagonal();

    // Step 3
    bS = P.row(n-1).asDiagonal() * bS * P.row(n-1).asDiagonal();
    bSWT = bS * W.row(n-1).transpose();
    ba(n-1) += W.row(n-1) * bSWT;
    bV.row(n-1).noalias() += W.row(n-1) * (bS + bS.transpose());
  }

  bU.row(0).setZero();
  ba(0) -= bV.row(0) * W.row(0).transpose();
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
void solve (
  const Eigen::MatrixBase<T1>& U,  // (N, J)
  const Eigen::MatrixBase<T2>& P,  // (N-1, J)
  const Eigen::MatrixBase<T3>& d,  // (N)
  const Eigen::MatrixBase<T1>& W,  // (N, J)
  Eigen::MatrixBase<T4>& Z,        // (N, Nrhs); initially set to Y
  Eigen::MatrixBase<T5>& F,        // (N, Nrhs); initially set to Y
  Eigen::MatrixBase<T5>& G         // (N, J*Nrhs)
) {
  int N = U.rows(), J = U.cols(), nrhs = Z.cols();

  Eigen::Matrix<typename T1::Scalar, T1::RowsAtCompileTime, T4::ColsAtCompileTime, T1::IsRowMajor> F_(J, nrhs);
  F_.setZero();
  F.row(0).setZero();

  for (int n = 1; n < N; ++n) {
    F_.noalias() += W.row(n-1).transpose() * Z.row(n-1);
    F_ = P.row(n-1).asDiagonal() * F_;
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < nrhs; ++k)
        F(n, j*nrhs+k) = F_(j, k);
    Z.row(n).noalias() -= U.row(n) * F_;
  }

  Z.array().colwise() /= d.array();

  F_.setZero();
  G.row(N-1).setZero();
  for (int n = N-2; n >= 0; --n) {
    F_.noalias() += U.row(n+1).transpose() * Z.row(n+1);
    F_ = P.row(n).asDiagonal() * F_;
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < nrhs; ++k)
        G(n, j*nrhs+k) = F_(j, k);
    Z.row(n).noalias() -= W.row(n) * F_;
  }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
void solve_grad (
  const Eigen::MatrixBase<T1>& U,  // (N, J)
  const Eigen::MatrixBase<T2>& P,  // (N-1, J)
  const Eigen::MatrixBase<T3>& d,  // (N)
  const Eigen::MatrixBase<T1>& W,  // (N, J)
  const Eigen::MatrixBase<T4>& Z,  // (N, Nrhs)
  const Eigen::MatrixBase<T5>& F,  // (N, J*Nrhs)
  const Eigen::MatrixBase<T5>& G,  // (N, J*Nrhs)
  const Eigen::MatrixBase<T4>& bZ, // (N, Nrhs)
  Eigen::MatrixBase<T6>& bU,       // (N, J)
  Eigen::MatrixBase<T7>& bP,       // (N-1, J)
  Eigen::MatrixBase<T8>& bd,       // (N)
  Eigen::MatrixBase<T6>& bW,       // (N, J)
  Eigen::MatrixBase<T9>& bY        // (N, Nrhs)
) {
  int N = U.rows(), J = U.cols(), nrhs = Z.cols();

  Eigen::Matrix<typename T4::Scalar, T4::RowsAtCompileTime, T4::ColsAtCompileTime, T4::IsRowMajor> Z_ = Z;
  typedef Eigen::Matrix<typename T1::Scalar, T1::RowsAtCompileTime, T4::ColsAtCompileTime, T1::IsRowMajor> F_t;
  F_t F_(J, nrhs), bF = F_t::Zero(J, nrhs);

  bY = bZ;
  for (int n = 0; n <= N-2; ++n) {
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < nrhs; ++k)
        F_(j, k) = G(n, j*nrhs+k);

    // Grad of: Z.row(n).noalias() -= W.row(n) * G;
    bW.row(n).noalias() -= bY.row(n) * F_.transpose();
    bF.noalias() -= W.row(n).transpose() * bY.row(n);

    // Inverse of: Z.row(n).noalias() -= W.row(n) * G;
    Z_.row(n).noalias() += W.row(n) * F_;

    // Grad of: g = P.row(n).asDiagonal() * G;
    bP.row(n).noalias() += (bF * F_.transpose()).diagonal();
    bF = P.row(n).asDiagonal() * bF;

    // Grad of: g.noalias() += U.row(n+1).transpose() * Z.row(n+1);
    bU.row(n+1).noalias() += Z_.row(n+1) * bF.transpose();
    bY.row(n+1).noalias() += U.row(n+1) * bF;
  }

  bY.array().colwise() /= d.array();
  bd.array() -= (Z_.array() * bY.array()).rowwise().sum();

  // Inverse of: Z.array().colwise() /= d.array();
  Z_.array().colwise() *= d.array();

  bF.setZero();
  for (int n = N-1; n >= 1; --n) {
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < nrhs; ++k)
        F_(j, k) = F(n, j*nrhs+k);

    // Grad of: Z.row(n).noalias() -= U.row(n) * f;
    bU.row(n).noalias() -= bY.row(n) * F_.transpose();
    bF.noalias() -= U.row(n).transpose() * bY.row(n);

    // Grad of: F = P.row(n-1).asDiagonal() * F;
    bP.row(n-1).noalias() += (bF * F_.transpose()).diagonal();
    bF = P.row(n-1).asDiagonal() * bF;

    // Grad of: F.noalias() += W.row(n-1).transpose() * Z.row(n-1);
    bW.row(n-1).noalias() += Z_.row(n-1) * bF.transpose();
    bY.row(n-1).noalias() += W.row(n-1) * bF;
  }
}

}

