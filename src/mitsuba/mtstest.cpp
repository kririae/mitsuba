#include <mitsuba/core/appender.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/gmm.h>
#include <mitsuba/core/platform.h>
#include <mitsuba/core/sched_remote.h>
#include <mitsuba/core/shvector.h>
#include <mitsuba/core/sshstream.h>
#include <mitsuba/core/sstream.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/core/version.h>
#include <mitsuba/render/renderjob.h>
#include <mitsuba/render/scenehandler.h>
#include <mitsuba/render/testcase.h>
#include <mitsuba/render/util.h>

#include <iostream>

using namespace mitsuba;

#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

/// https://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
/*
  We need a functor that can pretend it's const,
  but to be a good random number generator
  it needs mutable state.
*/
namespace Eigen {
namespace internal {
template <typename Scalar>
struct scalar_normal_dist_op {
  static boost::mt19937 rng;  // The uniform pseudo-random algorithm
  mutable boost::normal_distribution<Scalar> norm;  // The gaussian combinator

  EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)

  template <typename Index>
  inline const Scalar operator()(Index, Index = 0) const {
    return norm(rng);
  }
};

template <typename Scalar>
boost::mt19937 scalar_normal_dist_op<Scalar>::rng;

template <typename Scalar>
struct functor_traits<scalar_normal_dist_op<Scalar> > {
  enum {
    Cost         = 50 * NumTraits<Scalar>::MulCost,
    PacketAccess = false,
    IsRepeatable = false
  };
};
}  // end namespace internal
}  // end namespace Eigen

auto mtstest(int argc, char* argv[]) {
  int size = 2;           // Dimensionality (rows)
  int nn   = 1024 * 128;  // How many samples (columns) to draw
  Eigen::internal::scalar_normal_dist_op<double> randN;  // Gaussian functor
  Eigen::internal::scalar_normal_dist_op<double>::rng.seed();  // Seed the rng

  // Define mean and covariance of the distribution
  Eigen::VectorXd mean(size);
  Eigen::MatrixXd covar(size, size);

  mean << 1, 1;
  covar << 2, .5, .5, 1;

  Eigen::MatrixXd normTransform(size, size);

  Eigen::LLT<Eigen::MatrixXd> cholSolver(covar);

  // We can only use the cholesky decomposition if
  // the covariance matrix is symmetric, pos-definite.
  // But a covariance matrix might be pos-semi-definite.
  // In that case, we'll go to an EigenSolver
  if (cholSolver.info() == Eigen::Success) {
    // Use cholesky solver
    normTransform = cholSolver.matrixL();
  } else {
    // Use eigen solver
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
    normTransform = eigenSolver.eigenvectors() *
                    eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
  }

  Eigen::MatrixXd samples =
      (normTransform * Eigen::MatrixXd::NullaryExpr(size, nn, randN))
          .colwise() +
      mean;

  std::cout << "Target Mean\n" << mean << std::endl;
  std::cout << "Target Covar\n" << covar << std::endl;
  // std::cout << "Samples\n" << samples << std::endl;

  GMM::GaussianMixtureModel<2, 3> gmm;

  // stream samples
  for (int i = 0; i < nn; ++i) {
    const auto sample = samples.col(i);
    gmm.addBatchedSample(
        Vector2{static_cast<Float>(sample.x()), static_cast<Float>(sample.y())},
        1.0);
  }

  gmm.printInfo();

  return 0;
}

auto mtstest_main(int argc, char* argv[]) {
  /* Initialize the core framework */
  Class::staticInitialization();
  Object::staticInitialization();
  PluginManager::staticInitialization();
  Statistics::staticInitialization();
  Thread::staticInitialization();
  Logger::staticInitialization();
  FileStream::staticInitialization();
  Spectrum::staticInitialization();
  Bitmap::staticInitialization();
  Scheduler::staticInitialization();
  SHVector::staticInitialization();
  SceneHandler::staticInitialization();

  /* Correct number parsing on some locales (e.g. ru_RU) */
  setlocale(LC_NUMERIC, "C");

  int retval = mtstest(argc, argv);

  /* Shutdown the core framework */
  SceneHandler::staticShutdown();
  SHVector::staticShutdown();
  Scheduler::staticShutdown();
  Bitmap::staticShutdown();
  Spectrum::staticShutdown();
  FileStream::staticShutdown();
  Logger::staticShutdown();
  Thread::staticShutdown();
  Statistics::staticShutdown();
  PluginManager::staticShutdown();
  Object::staticShutdown();
  Class::staticShutdown();

  return retval;
}

#if !defined(__OSX__) && !defined(__WINDOWS__)
int main(int argc, char** argv) { return mtstest(argc, argv); }
#endif