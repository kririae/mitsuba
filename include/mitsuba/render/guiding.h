#pragma once
#if !defined(__MITSUBA_RENDER_GUIDING_H__)
#define __MITSUBA_RENDER_GUIDING_H__

#include <mitsuba/core/octree.h>
#include <mitsuba/core/shvector.h>
#include <mitsuba/core/util.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/scene.h>
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
                                     const Float inPdf, Point2& outUv,
                                     Float& outPdf) {
    const Vector3f& normalizedVector = normalize(inVector);

    // Calculate theta phi from solidAngle presented in Vector3f
    outUv = toSphericalCoordinates(normalizedVector);
    outUv.x /= M_PI;
    outUv.y /= (2 * M_PI);  // normalize to [0, 1]^2

    outPdf = inPdf * sin(outUv.x) * 2 * M_PI *
             M_PI;  // p(u, v) = 2 * \pi^2 * p(\theta, \phi)
  }

  FINLINE static void uvToSolidAngle(const Point2& inUv, const Float inPdf,
                                     Vector3f& outVector, Float& outPdf) {
    outVector = sphericalDirection(inUv.x * M_PI, inUv.y * 2 * M_PI);
    outPdf    = inPdf / (sin(inUv.x) * 2 * M_PI * M_PI);
  }
};

/**
 * @brief Configuration passing to \ref SpatialGuidingSampler
 */
struct SpatialGuidingConfig {
  SpatialGuidingConfig(Scene& scene) : scene(scene) {}
  SpatialGuidingConfig(Scene& scene, const Properties& prop) : scene(scene) {}

  /// Boolean configurations
  bool m_enabled{false};
  bool m_enable_photon_mapping{false};

  /// Numeric configurations
  std::size_t m_num_photons{0};

  Scene& scene;
};

/// Internal data used by \ref GuidingSample
struct GuidingSampleData {
  Spectrum weight;      //!< Raw contribution for the PDF
  Float    theta, phi;  //!< Spherical direction
};

/// The sample data structure used by both KD-Tree and GmmDistribution
struct GuidingSample {
  GuidingSample(GuidingSampleData& data) : data(data) {}
  ~GuidingSample() {}

  GuidingSampleData& data;
};

/* ==================================================================== */
/*                         Local Guiding Sampler                        */
/* ==================================================================== */
struct LocalGuidingSamplerBase {
  LocalGuidingSamplerBase() {}
  virtual ~LocalGuidingSamplerBase() {}

  virtual void addSample(const GuidingSample& sample) = 0;
  virtual void addSample(const std::vector<GuidingSample>& samples) {
    for (auto& sample : samples) addSample(sample);
  }

  virtual Float    pdf(const BSDFSamplingRecord& bRec) = 0;
  virtual Spectrum sample(BSDFSamplingRecord& bRec, Float& outPdf,
                          const Point2& sample)        = 0;
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

struct BSDFGuidingSampler : public LocalGuidingSamplerBase {
  /// The local sampler can be initialized with a series of samples
  /// accept nothing
  BSDFGuidingSampler(BSDFGuidingSamplerData& data)
      : LocalGuidingSamplerBase(), data(data) {}
  virtual ~BSDFGuidingSampler() {}

  /// Do nothing here
  virtual void addSample(const GuidingSample& sample) {}

  /// Calculate the PDF given a DirectionSamplingRecord from e.g., direct
  /// lighting
  virtual Float pdf(const BSDFSamplingRecord& bRec) {
    const auto& its  = data.its;
    const auto& bsdf = its.getBSDF();
    return bsdf == nullptr ? 0 : bsdf->pdf(bRec);
  }

  /// Provide exactly the same interface as BSDF
  virtual Spectrum sample(BSDFSamplingRecord& bRec, Float& outPdf,
                          const Point2& sample) {
    const auto& its  = data.its;
    const BSDF* bsdf = its.getBSDF();

    Spectrum bsdfWeight = bsdf->sample(bRec, outPdf, sample);
    return bsdfWeight;
  }

  BSDFGuidingSamplerData& data;
};

struct SHGuidingSamplerData {
  static constexpr int bands = 8;

  Intersection its;
  Sampler*     sampler;
  SHVector     shvec;

  // TODO: Factory function
  static FINLINE SHGuidingSamplerData
  MakeSHGuidingSamplerData(const Intersection& _its, Sampler* _sampler) {
    return SHGuidingSamplerData{
        .its = _its, .sampler = _sampler, .shvec = SHVector{bands}};
  }
};

/// This is not efficient in capturing high-frequency BSDF
struct SHGuidingSampler : public LocalGuidingSamplerBase {
  SHGuidingSampler(SHGuidingSamplerData& data)
      : LocalGuidingSamplerBase(), data(data) {
    data.shvec.addOffset(-data.shvec.findMinimum(16) + 0.1);
    data.shvec.normalize();
  }

  virtual ~SHGuidingSampler() {}

  /// Project the sample as a delta-distribution onto the shvec
  /// Note: the samples should be local sample
  virtual void addSample(const GuidingSample& sample) {
    auto& shvec = data.shvec;
    shvec.addDelta(sample.data.weight.getLuminance(), sample.data.theta,
                   sample.data.phi);
    shvec.normalize();
  }

  virtual void addSample(const std::vector<GuidingSample>& samples) {
    auto& shvec = data.shvec;
    for (auto& sample : samples)
      shvec.addDelta(sample.data.weight.getLuminance(), sample.data.theta,
                     sample.data.phi);
    shvec.normalize();
  }

  /// Calculate the PDF given a DirectionSamplingRecord from e.g., direct
  /// lighting.
  virtual Float pdf(const BSDFSamplingRecord& dRec) {
    // SLog(EError, "PDF cannot be acquired in SHGuidingSampler");
    // TODO: this method is not exactly correct
    auto& shvec = data.shvec;
    return shvec.eval(dRec.wo);
  }

  /// Provide exactly the same interface as bsdf->sample
  virtual Spectrum sample(BSDFSamplingRecord& bRec, Float& outPdf,
                          const Point2& sample) {
    const auto& its     = data.its;
    const BSDF* bsdf    = its.getBSDF();
    auto&       sampler = getSHSamplerInstance();
    Point2      sample_ = sample;
    // the output sample is [0, pi]x[0, 2pi]
    // but the output PDF is already in solidAngle
    outPdf = sampler.warp(data.shvec, sample_);

    // The transformed result lies on local coordinate
    bRec.wo = sphericalDirection(sample_.x, sample_.y);

    Spectrum result = bsdf->eval(bRec) / outPdf;  // already multiplied by cos
    return result;
  }

 protected:
  static inline SHSampler& getSHSamplerInstance() {
    /// why
    static ref<SHSampler> sampler =
        new SHSampler(SHGuidingSamplerData::bands, 12);
    return *sampler;
  }

 public:
  SHGuidingSamplerData& data;
};

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
  using STree            = DynamicOctree<GuidingSample>;
  using DTree            = DynamicOctree<LocalSamplerType>;

  /// Phases
  enum class EOpVariant {
    /// Training phase. Simply adding the samples
    EAddSample = 1,
    /// Eval phase. Build the distribution on-the-fly
    EAddToDist = 2,

    ESampleDist = EAddSample | EAddToDist
  };

  SpatialGuidingSampler(const SpatialGuidingConfig& cfg)
      : m_cfg(cfg),
        m_stree(cfg.scene.getAABB()),
        m_dtree(cfg.scene.getAABB()) {}
  virtual ~SpatialGuidingSampler() {}
  SpatialGuidingSampler(const SpatialGuidingSampler&)           = delete;
  SpatialGuidingSampler(SpatialGuidingSampler&&)                = delete;
  SpatialGuidingSampler operator=(const SpatialGuidingSampler&) = delete;

  /// Add samples without querying the DTree
  /// \see \ref EPhase
  void train() {
    m_op = EOpVariant::EAddSample;
    if (m_first_round) m_first_round = false;
  }

  /// Add samples and query the DTree
  /// \see \ref EPhase
  void eval() { m_op = EOpVariant::ESampleDist; }

  void addSample(const GuidingSample& sample) {
    if (m_op & EOpVariant::AddSample) {
    }
  }

  void addNSample(const std::vector<GuidingSample>& samples) {
    // TODO: use this naive implementation for now
    for (auto& sample : samples) addSample(sample);
  }

 private:
  const SpatialGuidingConfig& m_cfg;

  STree m_stree;
  DTree m_dtree;

  bool       m_first_round{true};
  EOpVariant m_op{EOpVariant::EAddSample};
};

MTS_NAMESPACE_END

#endif
