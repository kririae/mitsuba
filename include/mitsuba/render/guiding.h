#pragma once
#if !defined(__MITSUBA_RENDER_GUIDING_H__)
#define __MITSUBA_RENDER_GUIDING_H__

#include <mitsuba/core/kdtree.h>
#include <mitsuba/core/util.h>

MTS_NAMESPACE_BEGIN

/**
 * \brief These functions accept a sample of a random variable, converting it
 * into another r.v. and transform their corresponding PDFs.
 * TODO: implement "A Low Distortion Map Between Disk and Square"
 */
struct RandomVariableTransform {
  FINLINE static void solidAngleToUv(const Vector3f& inVector,
                                     const Float inPdf, Point2f& outUv,
                                     Float& outPdf) {
    const Vector3f& normalizedVector = normalize(inVector);

    // Calculate theta phi from solidAngle presented in Vector3f
    outUv = toSphericalCoordinates(normalizedVector);
    outUv.y /= (2 * M_PI);  // normalize to [0, 1]^2

    outPdf =
        inPdf * sin(outUv.x) * 2 * M_PI;  // p(u, v) = 2 * \pi * p(\theta, \phi)
  }

  FINLINE static void uvToSolidAngle(const Vector2f& inUv, const Float inPdf,
                                     Vector3f& outVector, Float& outPdf) {
    outVector = sphericalDirection(inUv.x, inUv.y * 2 * M_PI);
    outPdf    = inPdf / (sin(inUv.x) * 2 * M_PI);
  }
};

/**
 * @brief Configuration passing to \ref SpatialGuidingSampler
 */
struct SpatialGuidingConfig {
  SpatialGuidingConfig() {}
  SpatialGuidingConfig(const Properties& prop) {}

  /// Boolean configurations
  bool m_enabled{false};
  bool m_enable_photon_mapping{false};

  /// Numeric configurations
  std::size_t m_num_photons{0};
};

/// Internal data used by \ref GuidingSample
struct GuidingSampleData {
  Spectrum weight;      //!< Raw contribution for the PDF
  Float    theta, phi;  //!< Spherical direction
};

/// Internal data used by \ref GMM
struct GmmData {};

/// The sample data structure used by both KD-Tree and GmmDistribution
struct GuidingSample : public SimpleKDNode<Point3f, GuidingSampleData> {};

struct GMM : public SimpleKDNode<Point3f, GmmData> {};

/**
 * @brief The implementation of "On-line Learning of Parametric Mixture Models
 * for Light Transport Simulation"
 * It contains two phases,
 * 1. Training: Adding a series of samples, featuring f: (x, \omega): R6 ->
 *   Spectrum, we record the spectrum, and sample with respect to the spectrum
 *   average, i.e., construct the PDF from these samples
 * 2. Querying: Build distributions on-the-fly. For any query on point x, we
 *   find that if there's any existing nearby distributions(within a kernel),
 *   interpolate them(for GMM, re-weight the distributions) or select one. If
 *   there isn't any distribution, create a new one from existing samples.
 * 3. Add samples: we add samples on-the-fly, and the samples are added to
 *   adjacent distributions
 */
class SpatialGuidingSampler {
 public:
  using STree = PointKDTree<GuidingSample>;
  using DTree = PointKDTree<GMM>;

  SpatialGuidingSampler(const SpatialGuidingConfig& cfg) : m_cfg(cfg) {}
  virtual ~SpatialGuidingSampler() {}

  void train() {}
  void eval() {}

  void addSample(const GuidingSample& sample) {}
  void addNSample(const std::vector<GuidingSample>& samples) {}

 private:
  SpatialGuidingConfig m_cfg;

  STree m_stree;
  DTree m_dtree;
};

MTS_NAMESPACE_END

#endif
