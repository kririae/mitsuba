#pragma once
#if !defined(__MITSUBA_CORE_GMM_H__)
#define __MITSUBA_CORE_GMM_H__

#include <mitsuba/core/matrix.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/mitsuba.h>

#include <array>
#include <random>

MTS_NAMESPACE_BEGIN

namespace GMM {

#define GMM_CHECK(v)      \
  assert(!std::isnan(v)); \
  assert(!std::isinf(v));

template <int DDimension>
class GaussianSufficientStatistics;

template <>
class GaussianSufficientStatistics<2> {
 public:
  static auto sampleToMatrix(const Vector2& sample) {
    Matrix<DDimension, 1, Float> result;
    result.m[0][0] = sample[0];
    result.m[1][0] = sample[1];
    return result;
  }

  static auto sampleToTransposedMatrix(const Vector2& sample) {
    Matrix<1, DDimension, Float> result;
    result.m[0][0] = sample[0];
    result.m[0][1] = sample[1];
    return result;
  }

  static constexpr int DDimension = 2;
  using VectorType                = Vector2;
  using MatrixType                = Matrix2x2;

  GaussianSufficientStatistics() : first(1) {}
  GaussianSufficientStatistics(const VectorType& sample)
      : first(1),
        second(sample),
        third(sampleToMatrix(sample) * sampleToTransposedMatrix(sample)) {}
  GaussianSufficientStatistics(Float f, const VectorType& s,
                               const MatrixType& t)
      : first(f), second(s), third(t) {}

  Float      first{1};
  VectorType second;
  MatrixType third;

  inline GaussianSufficientStatistics operator+(
      const GaussianSufficientStatistics& other) const {
    return GaussianSufficientStatistics(
        first + other.first, second + other.second, third + other.third);
  }

  template <typename T>
  inline GaussianSufficientStatistics operator*(const T& value) const {
    return GaussianSufficientStatistics(first * value, second * value,
                                        third * value);
  }
};

template <int DDimension>  // the dimension
class MultivariateGaussian;

template <>
class MultivariateGaussian<2> {
 public:
  static constexpr int DDimension = 2;
  using VectorType                = Vector2;
  using MatrixType                = Matrix2x2;

  MultivariateGaussian() {
    m_mean = Vector2{0.0, 0.0};
    m_cov.setIdentity();
    bool cholres = m_cov.chol(m_chol);
    assert(cholres);
    m_cov.invert(m_icov);
    denom = sqrt(pow(2 * M_PI, DDimension) * m_cov.det());
  }

  MultivariateGaussian(const VectorType& mean, const MatrixType& cov)
      : m_mean(mean),
        m_cov(cov),
        denom(sqrt(pow(2 * M_PI, DDimension) * m_cov.det())) {
    m_cov.invert2x2(m_icov);
    bool cholres = m_cov.chol(m_chol);
    assert(cholres);
  }

  auto eval(const Vector2& s) -> Float {
    const auto delta = s - m_mean;
    auto       res   = -static_cast<Float>(0.5) * (m_icov * delta);
    return exp(dot(res, delta)) / denom;
  }

  auto sample() -> VectorType {
    thread_local static std::mt19937 rng;
    std::normal_distribution<Float>  norm;

    VectorType result;
    for (int i = 0; i < VectorType::dim; ++i) result[i] = norm(rng);
    return m_mean + m_chol * result;
  }

  auto getMean() const -> const VectorType& { return m_mean; }
  auto getCov() const -> const MatrixType& { return m_cov; }
  auto setMean(const VectorType& mean) { m_mean = mean; }
  auto setCov(const MatrixType& cov) {
    m_cov        = cov;
    bool cholres = m_cov.chol(m_chol);
    m_cov.invert2x2(m_icov);
    denom = sqrt(pow(2 * M_PI, DDimension) * m_cov.det());
    assert(cholres);
  }

  auto toString() const -> std::string {
    std::ostringstream oss;
    oss << "MultivariateGaussian[" << endl
        << "  DDimension = " << DDimension << "," << endl
        << "  m_mean = " << m_mean.toString() << "," << endl
        << "  m_conv = " << m_cov.toString() << endl
        << "]";
    return oss.str();
  }

 private:
  VectorType m_mean{};
  MatrixType m_cov, m_icov, m_chol;
  Float      denom;
};

struct EtaInstance {
  uint64_t i{1};
  Float    alpha{0.6};

  Float getEta() {
    const Float result = pow(i, -alpha);
    i++;
    return result;
  }
};

template <int DDimension, int KComponents>
class GaussianMixtureModel : public Object {
 public:
  using ModelType  = MultivariateGaussian<DDimension>;
  using StatType   = GaussianSufficientStatistics<DDimension>;
  using VectorType = typename ModelType::VectorType;
  using MatrixType = typename ModelType::MatrixType;

  GaussianMixtureModel() {
    for (int i = 0; i < KComponents; ++i) m_param_mix[i] = 1.0 / KComponents;
    std::inclusive_scan(m_param_mix.begin(), m_param_mix.end(),
                        m_param_mix_incl.begin());
  }

  static auto abT(const VectorType& a, const VectorType& b) -> MatrixType {
    return StatType::sampleToMatrix(a) * StatType::sampleToTransposedMatrix(b);
  }

  /* ==================================================================== */
  /*                           Sampler Interface                          */
  /* ==================================================================== */
  auto pdf(const VectorType& sample) -> Float {
    Float result = 0.0;
    for (int j = 0; j < KComponents; ++j)
      result += m_param_mix[j] * m_param_gauss[j].eval(sample);
    return result;
  }

  auto sample(const Vector2f& u, Float& outPdf) -> VectorType {
    /* Perform sampling on m_param_mix */
    auto lower =
        std::lower_bound(m_param_mix_incl.begin(), m_param_mix_incl.end(), u.x);
    const int index = lower - m_param_mix_incl.begin();
    assert(0 <= index && index < KComponents);
    /* Perform sampling on gaussian distribution */
    VectorType result = m_param_gauss[index].sample();
    outPdf            = pdf(result); /* marginal distribution */
    return result;
  }

  /* ==================================================================== */
  /*                             GMM Interface                            */
  /* ==================================================================== */

  /// return the un-normalized responsibility
  auto responsibility(const VectorType& sample, const int k /* component ID */)
      -> Float {
    assert(k < KComponents);
    GMM_CHECK(sample[0]);
    GMM_CHECK(sample[1]);
    GMM_CHECK(m_param_mix[k]);
    GMM_CHECK(m_param_gauss[k].eval(sample));
    return m_param_mix[k] * m_param_gauss[k].eval(sample);
  }

  /// (eq 7) in the original paper
  /// streaming E-step
  auto updateSufficientStats(const VectorType& sample, const Float& weight)
      -> void {
    const Float etaI = m_eta.getEta();
    GMM_CHECK(etaI);
    Float denom = 0.0;

    /* Normalized responsibility */
    std::array<Float, KComponents> pdf;
    std::fill(pdf.begin(), pdf.end(), 0);
    for (int j = 0; j < KComponents; ++j) {
      pdf[j] = responsibility(sample, j);
      denom += pdf[j];
    }

    if (denom == 0) {
      printf("[n=%ld,\n", m_eta.i);
      printf("sample=%s,\n", sample.toString().c_str());
      printf("mix=%f,\n", m_param_mix[0]);
      printf("gauss=%s]\n", m_param_gauss[0].toString().c_str());
    }

    denom = std::max<Float>(denom, 1e-5);
    for (int j = 0; j < KComponents; ++j) {
      auto& stat = m_stats[j];
      assert(stat.first != 0);
      stat = stat * (1 - etaI) +
             StatType(sample) * (etaI * weight * pdf[j] / denom); /* eq 7 */
      GMM_CHECK(etaI * weight * pdf[j] / denom);
      if (stat.first == 0) {
        printf("[n=%ld,\n", m_eta.i);
        printf("sample=%s,\n", sample.toString().c_str());
        printf("mix=%f,\n", m_param_mix[0]);
        printf("gauss=%s]\n", m_param_gauss[0].toString().c_str());
        assert(false);
      }
    }

    m_mixture_weight = (1 - etaI) * m_mixture_weight + etaI * weight;
  }

  /// (eq 9, 10)
  /// M-step
  auto updateModel() -> void {
    for (int j = 0; j < KComponents; ++j) {
      const auto& stat   = m_stats[j];
      const auto& first  = stat.first;
      const auto& second = stat.second;
      const auto& third  = stat.third;
      const auto& n      = m_eta.i;
      GMM_CHECK(first);
      if (first == 0) {
        std::cout << n << " " << j << std::endl;
        assert(false);
      }

      /* init A, B in the paper */
      MatrixType A, B;
      MatrixType I;
      I.setIdentity();

      /* The following parameters are written into two parts */
      auto mix = (first / m_mixture_weight) + (cp_v - 1) / n;
      mix /= (1 + KComponents * (cp_v - 1) / n);
      GMM_CHECK(mix);
      assert(mix != 0);

      auto mean = second / first;

      /* All 2x2 matrices */
      A = abT(second, mean) + abT(mean, second);
      B = abT(mean, mean);

      auto cov = (cp_b / n) * I + (third - A + first * B) / m_mixture_weight;
      cov /= (cp_a - 2) / n + first / m_mixture_weight;

      m_param_mix[j] = mix;
      m_param_gauss[j].setMean(mean);
      m_param_gauss[j].setCov(cov);
    }

    std::inclusive_scan(m_param_mix.begin(), m_param_mix.end(),
                        m_param_mix_incl.begin());
  }

  auto addBatchedSample(const VectorType& sample, const Float& weight) -> void {
    const auto& n = m_eta.i;
    updateSufficientStats(sample, weight);
    if ((n + 1) % m_freq == 0) updateModel();
  }

  /// debug port
  auto printInfo() -> void {
    printf("num_components: %d\n", KComponents);
    printf("[");
    for (int j = 0; j < KComponents; ++j) printf("%f ", m_param_mix[j]);
    printf("]\n");
    for (int j = 0; j < KComponents; ++j) {
      printf("%s\n", m_param_gauss[j].toString().c_str());
    }
  }

 private:
  /// These two represents the parameter vector
  std::array<ModelType, KComponents> m_param_gauss{};
  std::array<Float, KComponents>     m_param_mix{0};
  std::array<Float, KComponents>     m_param_mix_incl{0};
  std::array<StatType, KComponents>  m_stats{};

  EtaInstance m_eta{};
  Float       m_mixture_weight{1.0};

  int         batch_size = 32, m_freq = 4;
  const Float cp_a{2.01}, cp_b{5 * 1e-4}, cp_v{1.01};
};

MTS_NAMESPACE_END

}  // namespace GMM
#endif
