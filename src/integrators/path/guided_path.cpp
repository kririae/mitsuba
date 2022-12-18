#include <mitsuba/render/guiding.h>
#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/scene.h>

MTS_NAMESPACE_BEGIN

class GuidedPathIntegrator : public MonteCarloIntegrator {
 public:
  GuidedPathIntegrator(const Properties& props) : MonteCarloIntegrator(props) {}
  GuidedPathIntegrator(Stream* stream, InstanceManager* manager)
      : MonteCarloIntegrator(stream, manager) {
    Log(EError, "Network rendering is not implemented");
  }

  void serialize(Stream* stream, InstanceManager* manager) const {
    MonteCarloIntegrator::serialize(stream, manager);
    Log(EError, "Network rendering is not implemented");
  }

  std::string toString() const {
    std::ostringstream oss;
    oss << "GuidedPathIntegrator[" << endl
#if 0
        << "  maxDepth = " << m_maxDepth << "," << endl
        << "  rrDepth = " << m_rrDepth << "," << endl
        << "  strictNormals = " << m_strictNormals << endl
#endif
        << "]";
    return oss.str();
  }

  virtual ~GuidedPathIntegrator(){};
  void cancel() { MonteCarloIntegrator::cancel(); }

  /// Override the render function, execute it ourselves (because we want to add
  /// training phase and render phase)
  bool render(Scene* scene, RenderQueue* queue, const RenderJob* job,
              int sceneResID, int sensorResID, int samplerResID) {
    // Initializations
    ref<Scheduler> sched = Scheduler::getInstance();
    ref<Sensor> sensor = static_cast<Sensor*>(sched->getResource(sensorResID));
    ref<Film>   film   = sensor->getFilm();

    size_t         nCores = sched->getCoreCount();
    const Sampler* sampler =
        static_cast<const Sampler*>(sched->getResource(samplerResID, 0));
    size_t sampleCount = sampler->getSampleCount();

    Log(EInfo,
        "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SIZE_T_FMT
        " %s, " SSE_STR ") ..",
        film->getCropSize().x, film->getCropSize().y, sampleCount,
        sampleCount == 1 ? "sample" : "samples", nCores,
        nCores == 1 ? "core" : "cores");

    /* This is a sampling-based integrator - parallelize */
    ref<ParallelProcess> proc =
        new BlockedRenderProcess(job, queue, scene->getBlockSize());
    int integratorResID = sched->registerResource(this);
    proc->bindResource("integrator", integratorResID);
    proc->bindResource("scene", sceneResID);
    proc->bindResource("sensor", sensorResID);
    proc->bindResource("sampler", samplerResID);
    scene->bindUsedResources(proc);
    bindUsedResources(proc);
    sched->schedule(proc);

    m_process = proc;
    sched->wait(proc);
    m_process = NULL;
    sched->unregisterResource(integratorResID);

    return proc->getReturnStatus() == ParallelProcess::ESuccess;
  }

  /// Customized functions for guided path tracer
  Spectrum Li(const RayDifferential& r, RadianceQueryRecord& rRec) const {
    /* Some aliases and local variables */
    const Scene*    scene = rRec.scene;
    Intersection&   its   = rRec.its;
    RayDifferential ray(r);
    Spectrum        Li(0.0f);
    bool            scattered = false;

    /* Perform the first ray intersection (or ignore if the
       intersection has already been provided). */
    rRec.rayIntersect(ray);
    ray.mint = Epsilon;

    Spectrum throughput(1.0f);
    Float    eta = 1.0f;

    while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
      if (!its.isValid()) {
        /* If no intersection could be found, potentially return
           radiance from a environment luminaire if it exists */
        if ((rRec.type & RadianceQueryRecord::EEmittedRadiance) &&
            (!m_hideEmitters || scattered))
          Li += throughput * scene->evalEnvironment(ray);
        break;
      }

      const BSDF* bsdf = its.getBSDF(ray);

      /* Possibly include emitted radiance if requested */
      if (its.isEmitter() &&
          (rRec.type & RadianceQueryRecord::EEmittedRadiance) &&
          (!m_hideEmitters || scattered))
        Li += throughput * its.Le(-ray.d);

      /* Include radiance from a subsurface scattering model if requested */
      if (its.hasSubsurface() &&
          (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
        Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

      if ((rRec.depth >= m_maxDepth && m_maxDepth > 0) ||
          (m_strictNormals &&
           dot(ray.d, its.geoFrame.n) * Frame::cosTheta(its.wi) >= 0)) {
        /* Only continue if:
           1. The current path length is below the specifed maximum
           2. If 'strictNormals'=true, when the geometric and shading
              normals classify the incident direction to the same side */
        break;
      }

      /* ==================================================================== */
      /*                     Direct illumination sampling                     */
      /* ==================================================================== */

      /* Estimate the direct illumination if this is requested */
      DirectSamplingRecord dRec(its);
      // TODO: use BSDFSamplingRecord for now, then transform to
      // GuidingSamplingRecord
      // gSamplerData is to be acquired from KD-Tree
      auto gSamplerData =
          BSDFGuidingSamplerData::MakeBSDFGuidingSamplerData(its, rRec.sampler);
      auto gSampler = BSDFGuidingSampler(gSamplerData);

      if ((rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) &&
          (bsdf->getType() & BSDF::ESmooth)) {
        /* Perform direct *light* sampling */
        Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());

        if (!value.isZero()) {
          const Emitter* emitter = static_cast<const Emitter*>(dRec.object);

          /* Allocate a record for querying the BSDF */
          BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

          /* Evaluate BSDF * cos(theta) */
          const Spectrum bsdfVal = bsdf->eval(bRec);

          /* Prevent light leaks due to the use of shading normals */
          if (!bsdfVal.isZero() &&
              (!m_strictNormals ||
               dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {
            /* Calculate prob. of having generated that direction
               using importance sampling */
#if 0
            Float bsdfPdf =
                (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                    ? bsdf->pdf(bRec)
                    : 0;
#endif
            Float sPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                             ? gSampler.pdf(DirectionSamplingRecord{dRec.d})
                             : 0;

            /* Weight using the power heuristic */
            Float weight = miWeight(dRec.pdf, sPdf);
            Li += throughput * value * bsdfVal * weight;
          }
        }
      }

      /* ==================================================================== */
      /*                            BSDF sampling                             */
      /* ==================================================================== */

      /* Sample BSDF * cos(theta) */
      Float              bsdfPdf;
      BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
      // TODO: use BSDFSamplingRecord for now, then transform to
      // GuidingSamplingRecord
      Spectrum bsdfWeight = gSampler.sample(bRec, bsdfPdf, rRec.nextSample2D());
      // Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
      if (bsdfWeight.isZero()) break;

      scattered |= bRec.sampledType != BSDF::ENull;

      /* Prevent light leaks due to the use of shading normals */
      const Vector wo        = its.toWorld(bRec.wo);
      Float        woDotGeoN = dot(its.geoFrame.n, wo);
      if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0) break;

      bool     hitEmitter = false;
      Spectrum value;

      /* Trace a ray in this direction */
      ray = Ray(its.p, wo, ray.time);
      if (scene->rayIntersect(ray, its)) {
        /* Intersected something - check if it was a luminaire */
        if (its.isEmitter()) {
          value = its.Le(-ray.d);
          dRec.setQuery(ray, its);
          hitEmitter = true;
        }
      } else {
        /* Intersected nothing -- perhaps there is an environment map? */
        const Emitter* env = scene->getEnvironmentEmitter();

        if (env) {
          if (m_hideEmitters && !scattered) break;

          value = env->evalEnvironment(ray);
          if (!env->fillDirectSamplingRecord(dRec, ray)) break;
          hitEmitter = true;
        } else {
          break;
        }
      }

      /* Keep track of the throughput and relative
         refractive index along the path */
      throughput *= bsdfWeight;
      eta *= bRec.eta;

      /* If a luminaire was hit, estimate the local illumination and
         weight using the power heuristic */
      if (hitEmitter &&
          (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
        /* Compute the prob. of generating that direction using the
           implemented direct illumination sampling technique */
        const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta))
                                 ? scene->pdfEmitterDirect(dRec)
                                 : 0;
        /* Here bsdf/pdf is already merged into throughput. cos \theta' is
           merged into value. If this branch is not executed, its ok because it
           can be viewed as the estimator evaluates to zero */
        Li += throughput * value * miWeight(bsdfPdf, lumPdf);
      }

      /* ==================================================================== */
      /*                         Indirect illumination                        */
      /* ==================================================================== */

      /* Set the recursive query type. Stop if no surface was hit by the
         BSDF sample or if indirect illumination was not requested */
      if (!its.isValid() ||
          !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
        break;
      rRec.type = RadianceQueryRecord::ERadianceNoEmission;

      if (rRec.depth++ >= m_rrDepth) {
        /* Russian roulette: try to keep path weights equal to one,
           while accounting for the solid angle compression at refractive
           index boundaries. Stop with at least some probability to avoid
           getting stuck (e.g. due to total internal reflection) */

        Float q = std::min(throughput.max() * eta * eta, (Float)0.95f);
        if (rRec.nextSample1D() >= q) break;
        throughput /= q;
      }
    }

    return Li;
  }

  inline Float miWeight(Float pdfA, Float pdfB) const {
    pdfA *= pdfA;
    pdfB *= pdfB;
    return pdfA / (pdfA + pdfB);
  }

 private:
 public:
  MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_S(GuidedPathIntegrator, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(GuidedPathIntegrator, "Guided path integrator");
MTS_NAMESPACE_END
