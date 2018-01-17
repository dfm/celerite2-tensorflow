#include <Eigen/Core>

namespace celerite {

template <typename T1, typename T2, typename T3, typename T4, typename T5>
int factor (
  const Eigen::MatrixBase<T1>& U,  // (N, J)
  const Eigen::MatrixBase<T2>& P,  // (N-1, J)
  Eigen::MatrixBase<T3>& D,        // (N);    initially set to A
  Eigen::MatrixBase<T4>& W,        // (N, J); initially set to V
  Eigen::MatrixBase<T5>& S         // (J, J)
) {
  int N = U.rows(), J = U.cols();

  Eigen::Matrix<typename T4::Scalar, 1, T4::ColsAtCompileTime> tmp(1, J);

  // First row
  S.setZero();
  W.row(0) /= D(0);

  // The rest of the rows
  for (int n = 1; n < N; ++n) {
    // Update S = diag(P) * (S + D*W*W.T) * diag(P)
    S.noalias() += D(n-1) * W.row(n-1).transpose() * W.row(n-1);
    S.array() *= (P.row(n-1).transpose() * P.row(n-1)).array();

    // Update D = A - U * S * U.T
    tmp = U.row(n) * S;
    D(n) -= tmp * U.row(n).transpose();
    if (D(n) <= 0.0) return n;

    // Update W = (V - U * S) / D
    W.row(n).noalias() -= tmp;
    W.row(n) /= D(n);
  }

  return 0;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
void factor_grad (
  const Eigen::MatrixBase<T1>& U,   // (N, J)
  const Eigen::MatrixBase<T2>& P,   // (N-1, J)
  const Eigen::MatrixBase<T3>& D,   // (N)
  const Eigen::MatrixBase<T1>& W,   // (N, J)
  const Eigen::MatrixBase<T4>& S,   // (J, J)

  const Eigen::MatrixBase<T3>& bD,  // (N)
  const Eigen::MatrixBase<T1>& bW,  // (N, J)
  const Eigen::MatrixBase<T4>& bS,  // (J, J)

  Eigen::MatrixBase<T5>& bA,        // (N)
  Eigen::MatrixBase<T6>& bU,        // (N, J)
  Eigen::MatrixBase<T6>& bV,        // (N, J)
  Eigen::MatrixBase<T7>& bP         // (N-1, J)
) {
  int N = U.rows();

  // Make local copies of the gradients that we need.
  typename T3::Scalar bD_ = bD(N-1);
  Eigen::Matrix<typename T4::Scalar, T4::RowsAtCompileTime, T4::ColsAtCompileTime, T4::IsRowMajor> bS_ = bS, S_ = S;
  Eigen::Matrix<typename T4::Scalar, 1, T4::ColsAtCompileTime> bW_ = bW.row(N-1) / D(N-1);

  for (int n = N-1; n > 0; --n) {
    // Step 6
    bD_ -= W.row(n) * bW_.transpose();
    bV.row(n).noalias() = bW_;
    bU.row(n).noalias() = -bW_ * S_;
    bS_.noalias() -= U.row(n).transpose() * bW_;

    // Step 5
    bA(n) = bD_;
    bU.row(n).noalias() -= 2.0 * bD_ * U.row(n) * S_;
    bS_.noalias() -= bD_ * U.row(n).transpose() * U.row(n);

    // Step 4
    S_ *= P.row(n-1).asDiagonal().inverse();
    bP.row(n-1).noalias() = (bS_ * S_ + S_.transpose() * bS_).diagonal();

    // Step 3
    bS_ = P.row(n-1).asDiagonal() * bS_ * P.row(n-1).asDiagonal();
    bD_ = bD(n-1) + W.row(n-1) * bS_ * W.row(n-1).transpose();
    bW_ = bW.row(n-1) / D(n-1) + W.row(n-1) * (bS_ + bS_.transpose());

    // Downdate S
    S_ = P.row(n-1).asDiagonal().inverse() * S_;
    S_.noalias() -= D(n-1) * W.row(n-1).transpose() * W.row(n-1);
  }

  // Finally update the first row.
  bU.row(0).setZero();
  bV.row(0).noalias() = bW_;
  bD_ -= bW_ * W.row(0).transpose();
  bA(0) = bD_;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
void solve (
  const Eigen::MatrixBase<T1>& U,  // (N, J)
  const Eigen::MatrixBase<T2>& P,  // (N-1, J)
  const Eigen::MatrixBase<T3>& D,  // (N)
  const Eigen::MatrixBase<T1>& W,  // (N, J)
  Eigen::MatrixBase<T4>& Z,        // (N, Nrhs); initially set to Y
  Eigen::MatrixBase<T5>& f,        // (J, Nrhs)
  Eigen::MatrixBase<T5>& g         // (J, Nrhs)
) {
  int N = U.rows();

  f.setZero();
  g.setZero();

  for (int n = 1; n < N; ++n) {
    f.noalias() += W.row(n-1).transpose() * Z.row(n-1);
    f = P.row(n-1).asDiagonal() * f;
    Z.row(n).noalias() -= U.row(n) * f;
  }

  Z.array().colwise() /= D.array();

  for (int n = N-2; n >= 0; --n) {
    g.noalias() += U.row(n+1).transpose() * Z.row(n+1);
    g = P.row(n).asDiagonal() * g;
    Z.row(n).noalias() -= W.row(n) * g;
  }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
void solve_grad (
  const Eigen::MatrixBase<T1>& U,  // (N, J)
  const Eigen::MatrixBase<T2>& P,  // (N-1, J)
  const Eigen::MatrixBase<T3>& D,  // (N)
  const Eigen::MatrixBase<T1>& W,  // (N, J)
  const Eigen::MatrixBase<T4>& Z,  // (N, Nrhs)
  const Eigen::MatrixBase<T5>& f,  // (J, Nrhs)
  const Eigen::MatrixBase<T5>& g,  // (J, Nrhs)
  const Eigen::MatrixBase<T4>& bZ, // (N, Nrhs)
  const Eigen::MatrixBase<T5>& bf, // (J, Nrhs)
  const Eigen::MatrixBase<T5>& bg, // (J, Nrhs)
  Eigen::MatrixBase<T6>& bU,       // (N, J)
  Eigen::MatrixBase<T7>& bP,       // (N-1, J)
  Eigen::MatrixBase<T8>& bD,       // (N)
  Eigen::MatrixBase<T6>& bW,       // (N, J)
  Eigen::MatrixBase<T9>& bY        // (N, Nrhs)
) {
  int N = U.rows();

  Eigen::Matrix<typename T5::Scalar, T5::RowsAtCompileTime, T5::ColsAtCompileTime, T5::IsRowMajor>
    bf_ = bf, f_ = f, bg_ = bg, g_ = g;
  Eigen::Matrix<typename T4::Scalar, T4::RowsAtCompileTime, T4::ColsAtCompileTime, T4::IsRowMajor>
    Z_ = Z;

  bY = bZ;
  bU.row(0).setZero();

  for (int n = 0; n <= N-2; ++n) {
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

  for (int n = N-1; n >= 1; --n) {
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

}

