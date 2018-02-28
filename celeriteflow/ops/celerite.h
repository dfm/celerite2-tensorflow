#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace celerite {

template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename T1, typename T2, typename T3, typename T4>
int factor (
  const Eigen::MatrixBase<T1>& U,  // (N, J)
  const Eigen::MatrixBase<T2>& P,  // (N-1, J)
  Eigen::MatrixBase<T3>& d,        // (N);    initially set to A
  Eigen::MatrixBase<T4>& W         // (N, J); initially set to V
  //Eigen::MatrixBase<T5>& S         // (J, J)
) {
  int N = U.rows(), J = U.cols();

  Eigen::Matrix<typename T1::Scalar, 1, T1::ColsAtCompileTime> tmp(1, J);
  Eigen::Matrix<typename T1::Scalar, T1::ColsAtCompileTime, T1::ColsAtCompileTime> S(J, J);

  // First row
  S.setZero();
  W.row(0) /= d(0);

  // The rest of the rows
  for (int n = 1; n < N; ++n) {
    // Update S = diag(P) * (S + d*W*W.T) * diag(P)
    S.noalias() += d(n-1) * W.row(n-1).transpose() * W.row(n-1);
    S.array() *= (P.row(n-1).transpose() * P.row(n-1)).array();

    // Update d = a - U * S * U.T
    tmp = U.row(n) * S;
    d(n) -= tmp * U.row(n).transpose();
    if (d(n) <= 0.0) return n;

    // Update W = (V - U * S) / d
    W.row(n).noalias() -= tmp;
    W.row(n) /= d(n);
  }

  return 0;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
void factor_grad (
  const Eigen::MatrixBase<T1>& U,   // (N, J)
  const Eigen::MatrixBase<T2>& P,   // (N-1, J)
  const Eigen::MatrixBase<T3>& d,   // (N)
  const Eigen::MatrixBase<T1>& W,   // (N, J)

  Eigen::MatrixBase<T4>& bU,        // (N, J)
  Eigen::MatrixBase<T5>& bP,        // (N-1, J)
  Eigen::MatrixBase<T6>& ba,        // (N)
  Eigen::MatrixBase<T4>& bV         // (N, J)
) {
  int N = U.rows(), J = U.cols(), J2 = J*J;

  // Make local copies of the gradients that we need.
  typedef Eigen::Matrix<typename T1::Scalar, T1::ColsAtCompileTime, T1::ColsAtCompileTime, T1::IsRowMajor> S_t;
  S_t S_ = S_t::Zero(J, J), bS = S_t::Zero(J, J);
  Eigen::Matrix<typename T1::Scalar, T1::ColsAtCompileTime, 1> bSWT;

  // Pass through the data to compute all the S matrices
  const int Options = (T1::ColsAtCompileTime == Eigen::Dynamic) ? Eigen::Dynamic : T1::ColsAtCompileTime * T1::ColsAtCompileTime;
  Eigen::Matrix<typename T1::Scalar, T1::RowsAtCompileTime, Options, T1::IsRowMajor> S(N, J2);
  S.row(0).setZero();
  for (int n = 1; n < N; ++n) {
    S_.noalias() += d(n-1) * W.row(n-1).transpose() * W.row(n-1);
    S_.array() *= (P.row(n-1).transpose() * P.row(n-1)).array();
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < J; ++k)
        S(n, j*J+k) = S_(j, k);
  }

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

template <typename T1, typename T2, typename T3, typename T4>
void solve (
  const Eigen::MatrixBase<T1>& U,  // (N, J)
  const Eigen::MatrixBase<T2>& P,  // (N-1, J)
  const Eigen::MatrixBase<T3>& d,  // (N)
  const Eigen::MatrixBase<T1>& W,  // (N, J)
  Eigen::MatrixBase<T4>& Z         // (N, Nrhs); initially set to Y
) {
  int N = U.rows(), J = U.cols(), nrhs = Z.cols();

  Eigen::Matrix<typename T1::Scalar, T1::RowsAtCompileTime, T4::ColsAtCompileTime, T1::IsRowMajor> F(J, nrhs);
  F.setZero();

  for (int n = 1; n < N; ++n) {
    F.noalias() += W.row(n-1).transpose() * Z.row(n-1);
    F = P.row(n-1).asDiagonal() * F;
    Z.row(n).noalias() -= U.row(n) * F;
  }

  Z.array().colwise() /= d.array();

  F.setZero();
  for (int n = N-2; n >= 0; --n) {
    F.noalias() += U.row(n+1).transpose() * Z.row(n+1);
    F = P.row(n).asDiagonal() * F;
    Z.row(n).noalias() -= W.row(n) * F;
  }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
void solve_grad (
  const Eigen::MatrixBase<T1>& U,  // (N, J)
  const Eigen::MatrixBase<T2>& P,  // (N-1, J)
  const Eigen::MatrixBase<T3>& d,  // (N)
  const Eigen::MatrixBase<T1>& W,  // (N, J)
  const Eigen::MatrixBase<T4>& Z,  // (N, Nrhs)
  const Eigen::MatrixBase<T4>& bZ, // (N, Nrhs)
  Eigen::MatrixBase<T5>& bU,       // (N, J)
  Eigen::MatrixBase<T6>& bP,       // (N-1, J)
  Eigen::MatrixBase<T7>& bd,       // (N)
  Eigen::MatrixBase<T5>& bW,       // (N, J)
  Eigen::MatrixBase<T8>& bY        // (N, Nrhs)
) {
  int N = U.rows(), J = U.cols(), nrhs = Z.cols();

  Eigen::Matrix<typename T4::Scalar, T4::RowsAtCompileTime, T4::ColsAtCompileTime, T4::IsRowMajor> Z_ = Z;

  typedef Eigen::Matrix<typename T1::Scalar, T1::RowsAtCompileTime, T4::ColsAtCompileTime, T1::IsRowMajor> F_t;
  F_t bF = F_t::Zero(J, nrhs);

  std::vector<F_t> F(N);
  F[N-1].resize(J, nrhs);
  F[N-1].setZero();
  for (int n = N-2; n >= 0; --n) {
    F[n].resize(J, nrhs);
    F[n].noalias() = P.row(n).asDiagonal() * (F[n+1] + U.row(n+1).transpose() * Z_.row(n+1));
  }

  bY = bZ;

  for (int n = 0; n <= N-2; ++n) {
    // Grad of: Z.row(n).noalias() -= W.row(n) * G;
    bW.row(n).noalias() -= bY.row(n) * F[n].transpose();
    bF.noalias() -= W.row(n).transpose() * bY.row(n);

    // Inverse of: Z.row(n).noalias() -= W.row(n) * G;
    Z_.row(n).noalias() += W.row(n) * F[n];

    // Inverse of: G = P.row(n).asDiagonal() * G;
    //G_ = P.row(n).asDiagonal().inverse() * F[n];

    // Grad of: g = P.row(n).asDiagonal() * G;
    bP.row(n).noalias() += (bF * F[n].transpose()).diagonal();
    bF = P.row(n).asDiagonal() * bF;

    // Inverse of: g.noalias() += U.row(n+1).transpose() * Z.row(n+1);
    //G_.noalias() -= U.row(n+1).transpose() * Z_.row(n+1);

    // Grad of: g.noalias() += U.row(n+1).transpose() * Z.row(n+1);
    bU.row(n+1).noalias() += Z_.row(n+1) * bF.transpose();
    bY.row(n+1).noalias() += U.row(n+1) * bF;
  }

  bY.array().colwise() /= d.array();
  bd.array() -= (Z_.array() * bY.array()).rowwise().sum();

  // Inverse of: Z.array().colwise() /= d.array();
  Z_.array().colwise() *= d.array();

  F[0].setZero();
  for (int n = 1; n < N; ++n) {
    F[n].noalias() = P.row(n-1).asDiagonal() * (F[n-1] + W.row(n-1).transpose() * Z_.row(n-1));
  }

  bF.setZero();
  for (int n = N-1; n >= 1; --n) {
    // Grad of: Z.row(n).noalias() -= U.row(n) * f;
    bU.row(n).noalias() -= bY.row(n) * F[n].transpose();
    bF.noalias() -= U.row(n).transpose() * bY.row(n);

    // Inverse of: F = P.row(n-1).asDiagonal() * F;
    //F_ = P.row(n-1).asDiagonal().inverse() * F_;

    // Grad of: F = P.row(n-1).asDiagonal() * F;
    bP.row(n-1).noalias() += (bF * F[n].transpose()).diagonal();
    bF = P.row(n-1).asDiagonal() * bF;

    // Inverse of: F.noalias() += W.row(n-1).transpose() * Z.row(n-1);
    //F_.noalias() -= W.row(n-1).transpose() * Z_.row(n-1);

    // Grad of: F.noalias() += W.row(n-1).transpose() * Z.row(n-1);
    bW.row(n-1).noalias() += Z_.row(n-1) * bF.transpose();
    bY.row(n-1).noalias() += W.row(n-1) * bF;
  }
}

}

