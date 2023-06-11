#pragma once
#include "mitsuba/core/constants.h"
#if !defined(__MITSUBA_RENDER_GUIDING_H__)
#define __MITSUBA_RENDER_GUIDING_H__

#include <mitsuba/core/gmm.h>
#include <mitsuba/core/lock.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/octree.h>
#include <mitsuba/core/shvector.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/util.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/shape.h>

#include <memory_resource>
#include <mutex>
#include <optional>

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
  Float       m_search_radius{10.0};

  Scene& scene;

  std::string toString() const {
    std::ostringstream oss;
    oss << "SpatialGuidingConfig[" << endl
        << "  enabled = " << m_enabled << "," << endl
        << "  enable_photon_mapping = " << m_enable_photon_mapping << ","
        << endl
        << "  num_photons = " << m_num_photons << endl
        << "  search_radius = " << m_search_radius << endl
        << "]";
    return oss.str();
  }
};

/// The sample data structure used by both KD-Tree and GmmDistribution
struct GuidingSample {
  GuidingSample(const Float& weight, const Intersection& its,
                const Vector& localWo)
      : weight(weight) {
    const Point2& tp = toSphericalCoordinates(localWo);

    theta = tp.x;
    phi   = tp.y;
    GMM_CHECK(tp.x);
    GMM_CHECK(tp.y);
    position = its.p;
    normal   = its.geoFrame.n;
  }

  virtual ~GuidingSample() {}

  Float  weight;      //!< Raw contribution for the PDF
  Float  theta, phi;  //!< Spherical direction in *Local Coordinate*
  Point  position;    //!< Global Position
  Normal normal;      //!< Global Normal
};

/* ==================================================================== */
/*                         Local Guiding Sampler                        */
/* ==================================================================== */

/// Implementation of in this class should guarantee thread-safe
struct LocalGuidingSamplerBase {
  LocalGuidingSamplerBase(const Intersection& its, Sampler* sampler)
      : its(its), sampler(sampler) {}
  virtual ~LocalGuidingSamplerBase() {}

  virtual Normal getNormal() const { return its.geoFrame.n; }
  virtual Point  getPosition() const { return its.p; }

  virtual void addSample(const GuidingSample& sample) = 0;
  virtual void addSample(const std::vector<GuidingSample>& samples) {
    for (auto& sample : samples) addSample(sample);
  }

  virtual Float    pdf(const BSDFSamplingRecord& bRec) = 0;
  virtual Spectrum sample(BSDFSamplingRecord& bRec, Float& outPdf,
                          const Point2& sample)        = 0;

  Intersection its;
  Sampler*     sampler;
};

// Here no mutex is needed
struct BSDFGuidingSampler : public LocalGuidingSamplerBase {
  /// The local sampler can be initialized with a series of samples
  /// accept nothing
  BSDFGuidingSampler(const Intersection& its, Sampler* sampler)
      : LocalGuidingSamplerBase(its, sampler) {}
  virtual ~BSDFGuidingSampler() {}

  /// Do nothing here
  virtual void addSample(const GuidingSample& sample) {}

  /// Calculate the PDF given a DirectionSamplingRecord from e.g., direct
  /// lighting
  virtual Float pdf(const BSDFSamplingRecord& bRec) {
    const auto& bsdf = its.getBSDF();
    return bsdf == nullptr ? 0 : bsdf->pdf(bRec);
  }

  /// Provide exactly the same interface as BSDF
  virtual Spectrum sample(BSDFSamplingRecord& bRec, Float& outPdf,
                          const Point2& sample) {
    const BSDF* bsdf = its.getBSDF();

    Spectrum bsdfWeight = bsdf->sample(bRec, outPdf, sample);
    return bsdfWeight;
  }
};

/// This is not efficient in capturing high-frequency BSDF
struct SHGuidingSampler : public LocalGuidingSamplerBase {
  static constexpr int bands = 8;

  SHGuidingSampler(const Intersection& its, Sampler* sampler)
      : LocalGuidingSamplerBase(its, sampler), shvec(bands) {
    sample_mutex = new Mutex();
    shvec.addOffset(-shvec.findMinimum(16) + 0.1);
    shvec.normalize();
  }

  virtual ~SHGuidingSampler() {}

  /// Project the sample as a delta-distribution onto the shvec
  /// Note: the samples should be local sample
  virtual void addSample(const GuidingSample& sample) {
    LockGuard lock(sample_mutex);

    shvec.addDelta(sample.weight, sample.theta, sample.phi);
    shvec.normalize();
  }

  virtual void addSample(const std::vector<GuidingSample>& samples) {
    for (auto& sample : samples)
      shvec.addDelta(sample.weight, sample.theta, sample.phi);
  }

  /// Calculate the PDF given a DirectionSamplingRecord from e.g., direct
  /// lighting.
  virtual Float pdf(const BSDFSamplingRecord& dRec) {
    LockGuard lock(sample_mutex);

    // SLog(EError, "PDF cannot be acquired in SHGuidingSampler");
    // TODO: this method is not exactly correct
    return shvec.eval(dRec.wo);
  }

  /// Provide exactly the same interface as bsdf->sample
  virtual Spectrum sample(BSDFSamplingRecord& bRec, Float& outPdf,
                          const Point2& sample) {
    LockGuard lock(sample_mutex);

    const BSDF* bsdf    = its.getBSDF();
    auto&       sampler = getSHSamplerInstance();
    Point2      sample_ = sample;
    // the output sample is [0, pi]x[0, 2pi]
    // but the output PDF is already in solidAngle
    outPdf = sampler.warp(shvec, sample_);

    // The transformed result lies on local coordinate
    bRec.wo = sphericalDirection(sample_.x, sample_.y);

    Spectrum result = bsdf->eval(bRec) / outPdf;  // already multiplied by cos
    return result;
  }

 protected:
  static inline SHSampler& getSHSamplerInstance() {
    /// why
    static ref<SHSampler> sampler = new SHSampler(bands, 12);
    return *sampler;
  }

 public:
  SHVector   shvec;
  ref<Mutex> sample_mutex;
};

struct GMMGuidingSampler : public LocalGuidingSamplerBase {
  GMMGuidingSampler(const Intersection& its, Sampler* sampler)
      : LocalGuidingSamplerBase(its, sampler), sample_mutex(new Mutex()) {}
  virtual ~GMMGuidingSampler() {}

  virtual void addSample(const GuidingSample& sample) {
    LockGuard lock(sample_mutex);
    /* No area-preserving transformation for now */
    GMM_CHECK(sample.theta);
    GMM_CHECK(sample.phi);
    gmm.addBatchedSample(Vector2{sample.theta, sample.phi}, sample.weight);
  }

  virtual void addSample(const std::vector<GuidingSample>& samples) {
    LockGuard lock(sample_mutex);
    for (auto& sample : samples)
      gmm.addBatchedSample(Vector2{sample.theta, sample.phi}, sample.weight);
  }

  virtual Float pdf(const BSDFSamplingRecord& dRec) {
    LockGuard lock(sample_mutex);

    auto uv = toSphericalCoordinates(dRec.wo);
    return gmm.pdf(Vector2{uv.x, uv.y});
  }

  virtual Spectrum sample(BSDFSamplingRecord& bRec, Float& outPdf,
                          const Point2& sample) {
    LockGuard lock(sample_mutex);

    const BSDF* bsdf    = its.getBSDF();
    auto        sample_ = gmm.sample(Vector2{sample.x, sample.y}, outPdf);

    while (!(sample.x >= 0 && sample.x <= M_PI / 2 && sample_.y >= 0 &&
             sample_.y <= 2 * M_PI))
      sample_ = gmm.sample(Vector2{sample.x, sample.y}, outPdf);
    // outPdf    = outPdf / (abs(sin(sample_.x)) * 2 * M_PI * M_PI);
    assert(outPdf >= 0);

    // The transformed result lies on local coordinate
    bRec.wo = sphericalDirection(sample_.x, sample_.y);

    Spectrum result = bsdf->eval(bRec) / outPdf;  // already multiplied by cos
    return result;
  }

 public:
  ref<Mutex> sample_mutex;

  GMM::GaussianMixtureModel<2, 1> gmm;
};

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
class SpatialGuidingSampler : public Object {
 public:
  friend class GuidedPathIntegrator;

  using LocalSamplerType = _LocalSamplerType;

  using STree = UnsafeDynamicOctree<GuidingSample>;
  using DTree = UnsafeDynamicOctree<LocalSamplerType*>; /* store a pointer to
                                                           avoid copying */
  /// mem resource for storing samples
  std::pmr::monotonic_buffer_resource               m_mem_resource{};
  std::pmr::polymorphic_allocator<LocalSamplerType> allocator{&m_mem_resource};

  /// Phases
  enum EOpVariant {
    /// Training phase. Simply adding the samples
    EAddSample = 1,
    /// Eval phase. Build the distribution on-the-fly
    EAddToDist = 2,

    ESampleDist = EAddSample | EAddToDist
  };

  /// Initialize the SpatialGuidingSampler with scene.AABB
  SpatialGuidingSampler() = default;
  SpatialGuidingSampler(const SpatialGuidingConfig& cfg)
      : m_cfg(cfg),
        m_stree(cfg.scene.getAABB()),
        m_dtree(cfg.scene.getAABB()) {}
  virtual ~SpatialGuidingSampler() { m_mem_resource.release(); }
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

  /// Add a sample presented in local coordinate
  void addSample(const GuidingSample& sample) {
    bool hasDist = false;
    if (m_op & EOpVariant::EAddToDist) {
      // Search DTree to find adjacent distributions
      m_dtree.searchSphere(
          BSphere(sample.position, m_cfg.m_search_radius),
          [&](LocalSamplerType* lsampler) {
            if (distance(lsampler->getPosition(), sample.position) >
                m_cfg.m_search_radius)
              return;
            hasDist = true;
            lsampler->addSample(sample);
          });
    }

    if ((m_op & EOpVariant::EAddSample) && !hasDist) {
      // Add a single non-overlapping position
      m_stree.insert(sample, AABB{sample.position});
    }
  }

  void addNSample(const std::vector<GuidingSample>& samples) {
    // TODO: use this naive implementation for now
    for (auto& sample : samples) addSample(sample);
  }

  /**
   * @brief Acquire or build a LocalGuidingSampler from intersection and sampler
   * in params built using neighboring Samples or acquired from spatial cache.
   *
   * @param its
   * @param sampler
   * @return LocalGuidingSamplerBase*
   */
  std::optional<LocalSamplerType*> acquireSampler(const Intersection& its,
                                                  Sampler*            sampler) {
    const Point&  position      = its.p;
    const Normal& normal        = its.geoFrame.n;
    auto          valueFunction = [&](const LocalSamplerType* lsampler) {
      return pow(distance(position, lsampler->getPosition()), 2) /
                 m_cfg.m_search_radius +
             2 * sqrt(1 - std::min(1 - Epsilon,
                                            dot(normal, lsampler->getNormal())));
    };

    bool  hasSampler = false;
    Float value      = 1e9;

    LocalSamplerType* resultSampler{nullptr};
    auto              dTreeKernel = [&](LocalSamplerType* lsampler) {
      if (distance(lsampler->getPosition(), position) > m_cfg.m_search_radius)
        return;
      Float value_ = valueFunction(lsampler);
      assert(!std::isnan(value_));
      hasSampler = true;
      if (value_ < value) {
        value         = value_;
        resultSampler = lsampler;
      }
    };

    m_dtree.searchSphere(BSphere(position, m_cfg.m_search_radius), dTreeKernel);

    if (hasSampler) {
      // if there is neighbor sampler, return it
      assert(resultSampler != nullptr);
      return resultSampler;
    } else {
      // else, build a new local sampler here
      // create a new sampler
      auto* lsampler =
          allocator.template new_object<LocalSamplerType>(its, sampler);

      // add neighboring samples to this position
      bool hasSample   = false;
      auto sTreeKernel = [&](const GuidingSample& sample) {
        hasSample = true;
        lsampler->addSample(sample);
      };

      m_stree.searchSphere(BSphere(position, m_cfg.m_search_radius),
                           sTreeKernel);
      if (hasSample) {
        m_dtree.insert(lsampler, AABB{position});
        ++m_num_dist;
        // search the tree again to acquire the samplerData
        auto result = std::optional<LocalSamplerType*>(lsampler);
        assert(result.has_value());
        return result;
      } else {
        return std::nullopt;  // optional is empty
      }
    }
  }

  std::size_t getNumDistributions() const { return m_num_dist; }

 private:
  const SpatialGuidingConfig m_cfg;

  STree m_stree;
  DTree m_dtree;

  bool m_first_round{true};
  int  m_op{EOpVariant::ESampleDist};

  /// some statistics
  std::atomic<std::size_t> m_num_dist;
};

MTS_NAMESPACE_END

#endif
