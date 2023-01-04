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

auto mtstest(int argc, char* argv[]) {
  GMM::MultivariateGaussian<2> mvg;
  mvg.setMean(Vector2{1, 1});
  mvg.setCov(Matrix2x2{Vector2{1, .5}, Vector2{.5, 1}});

  std::cout << "Target Mean\n" << mvg.getMean().toString() << std::endl;
  std::cout << "Target Covar\n" << mvg.getCov().toString() << std::endl;

  GMM::GaussianMixtureModel<2, 1> gmm;

  // stream samples
  for (int i = 0; i < 1024 * 128; ++i) {
    const auto sample = mvg.sample();
    gmm.addBatchedSample(
        Vector2{static_cast<Float>(sample.x), static_cast<Float>(sample.y)},
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