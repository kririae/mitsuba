#pragma once
#if !defined(__MITSUBA_RENDER_GUIDING_H__)
#define __MITSUBA_RENDER_GUIDING_H__

#include <mitsuba/core/kdtree.h>
#include <mitsuba/core/util.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/shape.h>

#include <boost/variant.hpp>

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

/// The sample data structure used by both KD-Tree and GmmDistribution
struct GuidingSample : public SimpleKDNode<Point3f, GuidingSampleData> {};

struct LocalGuidingSamplerBase {
  LocalGuidingSamplerBase() {}
  virtual ~LocalGuidingSamplerBase() {}

  virtual void     addSample(const GuidingSample& sample)   = 0;
  virtual Float    pdf(const DirectionSamplingRecord& dRec) = 0;
  virtual Spectrum sample(BSDFSamplingRecord& dRec, Float& outPdf,
                          const Point2& sample)             = 0;
};

struct BSDFGuidingSamplerData {
  // boost::variant<Intersection, Normal> data;
  Intersection its;
  Sampler*     sampler;

  static FINLINE BSDFGuidingSamplerData
  MakeBSDFGuidingSamplerData(const Intersection& _its, Sampler* _sampler) {
    return BSDFGuidingSamplerData{.its = _its, .sampler = _sampler};
  }
};

struct BSDFGuidingSampler : public LocalGuidingSamplerBase,
                            SimpleKDNode<Point3, BSDFGuidingSamplerData> {
  /// The local sampler can be initialized with a series of samples
  /// Here accept nothing
  BSDFGuidingSampler(BSDFGuidingSamplerData& data)
      : LocalGuidingSamplerBase(), data(data) {}
  virtual ~BSDFGuidingSampler() {}

  /// Do nothing here
  virtual void addSample(const GuidingSample& sample) {}

  /// Calculate the PDF given a DirectionSamplingRecord from e.g., direct
  /// lighting
  virtual Float pdf(const DirectionSamplingRecord& dRec) {
    const auto& its  = data.its;
    const auto& bsdf = its.getBSDF();

    /* Allocate a record for querying the BSDF */
    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);
    return bsdf == nullptr ? 0 : bsdf->pdf(bRec);
  }

  /// Provide exactly the same interface as
  virtual Spectrum sample(BSDFSamplingRecord& bRec, Float& outPdf,
                          const Point2& sample) {
    const auto& its  = data.its;
    const BSDF* bsdf = its.getBSDF();

    Spectrum bsdfWeight = bsdf->sample(bRec, outPdf, sample);
    return bsdfWeight;
  }

  BSDFGuidingSamplerData& data;
};

struct SHGuidingSamplerData {};
struct SHGuidingSampler : public LocalGuidingSamplerBase,
                          SimpleKDNode<Point3, SHGuidingSamplerData> {};

#if 0
/// Internal data used by \ref GMM
struct GmmData {};

struct GMM : public SimpleKDNode<Point3f, GmmData>, LocalGuidingSamplerBase {};
#endif

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
template <typename _LocalSamplerType>
class SpatialGuidingSampler {
 public:
  using LocalSamplerType = _LocalSamplerType;
  using STree            = PointKDTree<GuidingSample>;
  using DTree            = PointKDTree<LocalSamplerType>;

  /// Phases
  enum class EOpVariant {
    /// Training phase. Simply adding the samples
    EAddSample = 1,
    /// Eval phase. Build the distribution on-the-fly
    EAddToDist = 2,

    ESampleDist = EAddSample | EAddToDist
  };

  SpatialGuidingSampler(const SpatialGuidingConfig& cfg) : m_cfg(cfg) {}
  virtual ~SpatialGuidingSampler() {}

  /// Add samples without querying the DTree
  /// \see \ref EPhase
  void train() {
    m_op = EOpVariant::EAddSample;
    if (m_first_round) m_first_round = false;
  }

  /// Add samples and query the DTree
  /// \see \ref EPhase
  void eval() { m_op = EOpVariant::ESampleDist; }

  void addSample(const GuidingSample& sample) {}
  void addNSample(const std::vector<GuidingSample>& samples) {}

 private:
  SpatialGuidingConfig m_cfg{};

  STree m_stree{};
  DTree m_dtree{};

  bool       m_first_round{true};
  EOpVariant m_op{EOpVariant::EAddSample};
};

MTS_NAMESPACE_END

#endif
