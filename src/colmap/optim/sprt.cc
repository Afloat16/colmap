// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/sprt.h"

#include <cmath>

namespace colmap {

SPRT::SPRT(const Options& options) { Update(options); }

void SPRT::Update(const Options& options) {
  options_ = options;
  log_delta_epsilon_ = std::log(options.delta) - std::log(options.epsilon);
  log_delta_1_epsilon_1_ =
      std::log1p(-options.delta) - std::log1p(-options.epsilon);
  UpdateDecisionThreshold();
}

bool SPRT::Evaluate(const std::vector<double>& residuals,
                    const double max_residual,
                    size_t* num_inliers,
                    size_t* num_eval_samples) {
  *num_inliers = 0;

  // Accumulate in log space so a long inlier prefix cannot underflow the
  // likelihood ratio and prevent rejection of a later outlier suffix.
  double log_likelihood_ratio = 0;

  for (size_t i = 0; i < residuals.size(); ++i) {
    if (std::abs(residuals[i]) <= max_residual) {
      *num_inliers += 1;
      log_likelihood_ratio += log_delta_epsilon_;
    } else {
      log_likelihood_ratio += log_delta_1_epsilon_1_;
    }

    if (log_likelihood_ratio > log_decision_threshold_) {
      *num_eval_samples = i + 1;
      return false;
    }
  }

  *num_eval_samples = residuals.size();

  return true;
}

void SPRT::UpdateDecisionThreshold() {
  // Equation 2
  const double C = (1 - options_.delta) *
                       std::log((1 - options_.delta) / (1 - options_.epsilon)) +
                   options_.delta * std::log(options_.delta / options_.epsilon);

  // Equation 6
  const double A0 =
      options_.eval_time_ratio * C / options_.num_models_per_sample + 1;

  double A = A0;

  const double kEps = 1.5e-8;

  // Compute A using the recursive relation
  //    A* = lim(n->inf) A
  // The series typically converges within 4 iterations

  for (size_t i = 0; i < 100; ++i) {
    const double A1 = A0 + std::log(A);

    if (std::abs(A1 - A) < kEps) {
      break;
    }

    A = A1;
  }

  log_decision_threshold_ = std::log(A);
}

}  // namespace colmap
