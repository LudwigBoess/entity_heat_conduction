#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/numeric.h"

#include "traits/pgen.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "framework/containers/particles.h"
#include "framework/domain/metadomain.h"
#include "kernels/particle_moments.hpp"
#include "kernels/particle_shapes.hpp"
#include "kernels/pushers/context.h"

#if defined(MPI_ENABLED)
  #include "arch/mpi_aliases.h"
  #include <mpi.h>
#endif

#include <cmath>

// helper: reconstruct a continuous cell coordinate Xi from (cell index, displacement)
#define i_di_to_Xi(I, DI) (static_cast<real_t>((I)) + static_cast<real_t>((DI)))

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    // Background magnetic field; returned unit vector is in units of B0 (=1/larmor0).
    InitFields(real_t btheta, real_t bphi)
      : Btheta { btheta * static_cast<real_t>(convert::deg2rad) }
      , Bphi { bphi * static_cast<real_t>(convert::deg2rad) } {}

    Inline auto bx1(const coord_t<D>&) const -> real_t {
      return math::cos(Btheta);
    }
    Inline auto bx2(const coord_t<D>&) const -> real_t {
      return math::sin(Btheta) * math::sin(Bphi);
    }
    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return math::sin(Btheta) * math::cos(Bphi);
    }
    Inline auto ex1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }
    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }
    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t Btheta, Bphi;
  };

  template <Dimension D>
  struct ConstantDensity {
    ConstantDensity() {}
    Inline auto operator()(const coord_t<D>&) const -> real_t {
      return ONE;
    }
  };

  /*
    Solve for the cold-component parallel drift vd that gives zero net current
    <v_par>=0 for the two-component distribution (Roberg-Clark et al. 2018):
      vd = vTh/sqrt(pi) - vTc*exp(-(vd/vTc)^2) / (sqrt(pi)*(1 + erf(vd/vTc))).
    Monotonic in vd on [0, vTh]; solved by bisection (host side).
  */
  inline auto solve_vd(real_t vTh, real_t vTc) -> real_t {
    const real_t inv_sqrtpi = static_cast<real_t>(0.5641895835477563);
    real_t       lo = ZERO, hi = vTh;
    for (int it = 0; it < 80; ++it) {
      const real_t mid = static_cast<real_t>(0.5) * (lo + hi);
      const real_t a   = mid / vTc;
      const real_t rhs = vTh * inv_sqrtpi -
                         vTc * std::exp(-a * a) * inv_sqrtpi / (ONE + std::erf(a));
      if (mid - rhs > ZERO) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    return static_cast<real_t>(0.5) * (lo + hi);
  }

  /*
    Invert the cold-component mean parallel velocity for the re-injection drift vd:
    find vd >= 0 such that <v_par>_c(vd) = m_target, where
      <v_par>_c(vd) = -vTc*exp(-(vd/vTc)^2)/(sqrt(pi)*(1+erf(vd/vTc))) - vd.
    Used by CustomPostStep so the re-injected cold-reservoir current cancels the
    outgoing current (m_target = -<v_par,out>). (Host side.)
  */
  inline auto invert_cold_mean(real_t m_target, real_t vTc) -> real_t {
    const real_t inv_sqrtpi = static_cast<real_t>(0.5641895835477563);
    if (m_target >= -vTc * inv_sqrtpi) {
      return ZERO; // would need vd < 0
    }
    real_t lo = ZERO, hi = static_cast<real_t>(8.0) * vTc - m_target;
    for (int it = 0; it < 80; ++it) {
      const real_t mid = static_cast<real_t>(0.5) * (lo + hi);
      const real_t a   = mid / vTc;
      const real_t mc  = -vTc * std::exp(-a * a) * inv_sqrtpi / (ONE + std::erf(a)) - mid;
      if (mc - m_target > ZERO) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    return static_cast<real_t>(0.5) * (lo + hi);
  }

  /*
    Two-component heat-flux-carrying distribution (Roberg-Clark et al. 2018, Eq.1):
    a HOT isotropic half-Maxwellian (v_par>0, streaming +x) plus a COLD isotropic
    half-Maxwellian (v_par<0, streaming -x) drifted by -vd, each density n0/2; vd
    enforces zero net current. v_par along x (|| B0). Carries q_par = q0.
  */
  template <Dimension D>
  struct RC_edist {
    RC_edist(random_number_pool_t& pool, real_t T_hot, real_t T_cold, real_t vd)
      : pool { pool }
      , T_hot { T_hot }
      , T_cold { T_cold }
      , vd { vd } {}

    Inline void operator()(const coord_t<D>& x_Ph, vec_t<Dim::_3D>& v) const {
      auto rand_gen = pool.get_state();
      auto xi       = Random<real_t>(rand_gen);
      pool.free_state(rand_gen);

      if (xi < static_cast<real_t>(0.5)) {
        // hot half-Maxwellian, v_par > 0 (streams +x)
        arch::energy_dist::JuttnerSinge(v, T_hot, pool);
        v[0] = math::abs(v[0]);
      } else {
        // cold drifted half-Maxwellian, v_par < 0 (streams -x, drift center -vd)
        int guard = 0;
        do {
          arch::energy_dist::JuttnerSinge(v, T_cold, pool);
          v[0] = v[0] - vd;
          ++guard;
        } while (v[0] > ZERO && guard < 32);
      }
    }

    random_number_pool_t pool;
    real_t               T_hot, T_cold, vd;
  };

  /*
    Heat-flux third moment q_dir = sum_p (1/2) m (v_dir) v^2, deposited with the
    same S-shape as the standard particle moments (N, T). dir in {1,2,3} selects
    q_x / q_y / q_z. Lab-frame third moment; since the bulk drift is ~0 here it
    equals the conductive heat flux (q_|| = q_x, along B0). Cartesian (Minkowski).
  */
  template <SimEngine::type S, class M, uint8_t N>
  class HeatFluxMoment_kernel {
    static constexpr auto D = M::Dim;

    const uint8_t                 dir;
    scatter_ndfield_t<D, N>       Buff;
    const idx_t                   buff_idx;
    const ntt::ParticleArrays     particles;
    const float                   mass;
    const bool                    use_weights;
    const M                       metric;
    const real_t                  inv_n0;
    const uint8_t                 order, window;
    const OutputSmoothingTypeFlag smoothing;

  public:
    HeatFluxMoment_kernel(uint8_t                                dir,
                          const scatter_ndfield_t<D, N>&         scatter_buff,
                          idx_t                                  buff_idx,
                          const Particles<M::Dim, M::CoordType>& prtls,
                          bool                                   use_weights,
                          const M&                               metric,
                          real_t                                 inv_n0,
                          uint8_t                                order,
                          OutputSmoothingTypeFlag                smoothing)
      : dir { dir }
      , Buff { scatter_buff }
      , buff_idx { buff_idx }
      , particles { static_cast<const ntt::ParticleArrays&>(prtls) }
      , mass { prtls.mass() }
      , use_weights { use_weights }
      , metric { metric }
      , inv_n0 { inv_n0 }
      , order { order }
      , window { static_cast<uint8_t>(math::ceil(static_cast<float>(order) / 2.0f)) }
      , smoothing { smoothing } {}

    Inline auto shapeFunction(real_t delta_x) const -> real_t {
      if (smoothing == OutputSmoothingType::SPLINE) {
        if (order == 0) {
          return ONE;
        } else if (order == 1) {
          return prtl_shape::S1(delta_x);
        } else if (order == 2) {
          return prtl_shape::S2(delta_x);
        } else if (order == 3) {
          return prtl_shape::S3(delta_x);
        } else {
          return ZERO;
        }
      } else {
        return ONE / (TWO * static_cast<real_t>(window) + ONE);
      }
    }

    Inline void operator()(prtlidx_t p) const {
      if (particles.tag(p) == ParticleTag::dead) {
        return;
      }
      const real_t ux    = particles.ux1(p);
      const real_t uy    = particles.ux2(p);
      const real_t uz    = particles.ux3(p);
      const real_t u2    = ux * ux + uy * uy + uz * uz;
      const real_t gamma = math::sqrt(ONE + u2);
      const real_t v2    = u2 / (gamma * gamma);
      const real_t u_dir = (dir == 1) ? ux : ((dir == 2) ? uy : uz);
      real_t       coeff = HALF * static_cast<real_t>(mass) * (u_dir / gamma) * v2;
      if constexpr (D == Dim::_1D) {
        coeff *= inv_n0 / metric.sqrt_det_h(
                            { static_cast<real_t>(particles.i1(p)) + HALF });
      } else if constexpr (D == Dim::_2D) {
        coeff *= inv_n0 / metric.sqrt_det_h(
                            { static_cast<real_t>(particles.i1(p)) + HALF,
                              static_cast<real_t>(particles.i2(p)) + HALF });
      } else {
        coeff *= inv_n0 / metric.sqrt_det_h(
                            { static_cast<real_t>(particles.i1(p)) + HALF,
                              static_cast<real_t>(particles.i2(p)) + HALF,
                              static_cast<real_t>(particles.i3(p)) + HALF });
      }
      if (use_weights) {
        coeff *= particles.weight(p);
      }

      auto buff_access = Buff.access();
      if constexpr (D == Dim::_1D) {
        for (int di1 = -window; di1 <= window; ++di1) {
          const real_t dl1 = math::abs(static_cast<real_t>(particles.dx1(p)) -
                                       (static_cast<real_t>(di1) + HALF));
          buff_access(particles.i1(p) + di1 + N_GHOSTS, buff_idx) +=
            coeff * shapeFunction(dl1);
        }
      } else if constexpr (D == Dim::_2D) {
        for (int di2 = -window; di2 <= window; ++di2) {
          for (int di1 = -window; di1 <= window; ++di1) {
            const real_t dl1 = math::abs(static_cast<real_t>(particles.dx1(p)) -
                                         (static_cast<real_t>(di1) + HALF));
            const real_t dl2 = math::abs(static_cast<real_t>(particles.dx2(p)) -
                                         (static_cast<real_t>(di2) + HALF));
            buff_access(particles.i1(p) + di1 + N_GHOSTS,
                        particles.i2(p) + di2 + N_GHOSTS,
                        buff_idx) += coeff * shapeFunction(dl1) * shapeFunction(dl2);
          }
        }
      } else {
        for (int di3 = -window; di3 <= window; ++di3) {
          for (int di2 = -window; di2 <= window; ++di2) {
            for (int di1 = -window; di1 <= window; ++di1) {
              const real_t dl1 = math::abs(static_cast<real_t>(particles.dx1(p)) -
                                           (static_cast<real_t>(di1) + HALF));
              const real_t dl2 = math::abs(static_cast<real_t>(particles.dx2(p)) -
                                           (static_cast<real_t>(di2) + HALF));
              const real_t dl3 = math::abs(static_cast<real_t>(particles.dx3(p)) -
                                           (static_cast<real_t>(di3) + HALF));
              buff_access(particles.i1(p) + di1 + N_GHOSTS,
                          particles.i2(p) + di2 + N_GHOSTS,
                          particles.i3(p) + di3 + N_GHOSTS,
                          buff_idx) += coeff * shapeFunction(dl1) *
                                       shapeFunction(dl2) * shapeFunction(dl3);
            }
          }
        }
      }
    }
  };

  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };
    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_2D, Dim::_3D> {}
    };

    const SimulationParams& params;

    const real_t     global_xmin, global_xmax;
    const real_t     T_hot, temperature_ratio, T_cold, T_ions;
    real_t           vd;          // cold re-injection drift (dynamic; CustomPostStep)
    array_t<real_t*> bdry_accum;  // [0]=sum outgoing v_par at cold reservoir, [1]=count
    const real_t     Btheta, Bphi;
    InitFields<D>    init_flds;

    PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : params { p }
      , global_xmin { m.mesh().extent(in::x1).first }
      , global_xmax { m.mesh().extent(in::x1).second }
      , T_hot { p.template get<real_t>("setup.T_hot") }
      , temperature_ratio { p.template get<real_t>("setup.temperature_ratio") }
      , T_cold { T_hot / temperature_ratio }
      , T_ions { p.template get<real_t>("setup.T_ions", ZERO) }
      , vd { solve_vd(std::sqrt(TWO * T_hot), std::sqrt(TWO * T_cold)) }
      , bdry_accum { "rc_bdry_accum", 2 }
      , Btheta { p.template get<real_t>("setup.Btheta", ZERO) }
      , Bphi { p.template get<real_t>("setup.Bphi", ZERO) }
      , init_flds { Btheta, Bphi } {}

    inline void InitPrtls(Domain<S, M>& domain) {
      // background ions: stationary, charge-neutralizing (pusher = "None")
      const auto T_p          = T_ions / domain.species[1].mass();
      const auto maxwellian_p = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        domain.random_pool(),
        T_p);

      // two-component heat-flux distribution for the electrons (mass = 1)
      const auto electron_dist = RC_edist<M::Dim>(domain.random_pool(),
                                                  T_hot,
                                                  T_cold,
                                                  vd);
      // uniform density (heat flux is carried in velocity space, not by a gradient)
      const auto density = ConstantDensity<M::Dim>();

      arch::InjectNonUniform<S, M, decltype(electron_dist), decltype(maxwellian_p), decltype(density)>(
        params,
        domain,
        { 1, 2 },
        { electron_dist, maxwellian_p },
        density,
        ONE);
    }

    /*
      Thermal-reservoir re-injection at the two x-boundaries (Roberg-Clark 2018).
      A particle leaving x=xmin (hot) is re-injected from f_h (v_par>0); one leaving
      x=xmax (cold) from f_c (v_par<0, drift -vd). vd for the cold reservoir is
      recalculated each step by CustomPostStep so the re-injected current cancels
      the outgoing current; the outgoing v_par is accumulated into bdry_accum here.
    */
    struct CustomPrtlUpdate {
      random_number_pool_t pool;
      real_t               T_hot, T_cold, vd;
      real_t               xmin, xmax;
      array_t<real_t*>     bdry_accum;

      CustomPrtlUpdate(random_number_pool_t&   pool,
                       real_t                  T_hot,
                       real_t                  T_cold,
                       real_t                  vd,
                       real_t                  xmin,
                       real_t                  xmax,
                       const array_t<real_t*>& bdry_accum)
        : pool { pool }
        , T_hot { T_hot }
        , T_cold { T_cold }
        , vd { vd }
        , xmin { xmin }
        , xmax { xmax }
        , bdry_accum { bdry_accum } {}

      Inline void operator()(prtlidx_t                        p,
                             const kernel::sr::PusherContext& ctx,
                             const kernel::sr::PusherBoundaries<M::Dim>&,
                             const ntt::ParticleArrays& particles,
                             const M&                   metric) const {

        const auto x_Cd = static_cast<real_t>(particles.i1(p)) +
                          static_cast<real_t>(particles.dx1(p));
        const auto x_Ph = metric.template convert<1, Crd::Cd, Crd::XYZ>(x_Cd);
        vec_t<Dim::_3D>       v { ZERO };
        const coord_t<M::Dim> x_dummy { ZERO };

        const real_t gamma_p = math::sqrt(ONE + SQR(particles.ux1(p)) +
                                          SQR(particles.ux2(p)) +
                                          SQR(particles.ux3(p)));
        const real_t beta_x_p = math::abs(particles.ux1(p)) / gamma_p;

        // hot reservoir at the left boundary: re-inject from f_h (v_par > 0)
        if (x_Ph < xmin) {
          arch::energy_dist::JuttnerSinge(v, T_hot, pool);
          v[0] = math::abs(v[0]);

          const int      delta_i1_to_wall  = particles.i1_prev(p);
          const prtldx_t delta_dx1_to_wall = particles.dx1_prev(p);
          const real_t   dx_to_wall = i_di_to_Xi(delta_i1_to_wall, delta_dx1_to_wall);
          const real_t   dt_to_wall = dx_to_wall /
                                    metric.template transform<1, Idx::XYZ, Idx::U>(
                                      x_dummy,
                                      beta_x_p);

          particles.ux1(p) = v[0];
          particles.ux2(p) = v[1];
          particles.ux3(p) = v[2];

          const real_t remaining_dt            = ctx.dt - dt_to_wall;
          const real_t remaining_dt_inv_energy = remaining_dt /
                                                 math::sqrt(
                                                   ONE + SQR(particles.ux1(p)) +
                                                   SQR(particles.ux2(p)) +
                                                   SQR(particles.ux3(p)));

          particles.i1(p)  = 0;
          particles.dx1(p) = metric.template transform<1, Idx::XYZ, Idx::U>(
                               x_dummy,
                               particles.ux1(p)) *
                             remaining_dt_inv_energy;

        } else if (x_Ph > xmax) {
          // cold reservoir: record outgoing current (v_par>0), then re-inject f_c
          Kokkos::atomic_add(&bdry_accum(0), beta_x_p);
          Kokkos::atomic_add(&bdry_accum(1), ONE);

          int guard = 0;
          do {
            arch::energy_dist::JuttnerSinge(v, T_cold, pool);
            v[0] = v[0] - vd;
            ++guard;
          } while (v[0] > ZERO && guard < 32);

          const int      delta_i1_to_wall  = ctx.ni1 - 1 - particles.i1_prev(p);
          const prtldx_t delta_dx1_to_wall = ONE - particles.dx1_prev(p);
          const real_t   dx_to_wall = i_di_to_Xi(delta_i1_to_wall, delta_dx1_to_wall);
          const real_t   dt_to_wall = dx_to_wall /
                                    metric.template transform<1, Idx::XYZ, Idx::U>(
                                      x_dummy,
                                      beta_x_p);

          particles.ux1(p) = v[0];
          particles.ux2(p) = v[1];
          particles.ux3(p) = v[2];

          const real_t remaining_dt            = ctx.dt - dt_to_wall;
          const real_t remaining_dt_inv_energy = remaining_dt /
                                                 math::sqrt(
                                                   ONE + SQR(particles.ux1(p)) +
                                                   SQR(particles.ux2(p)) +
                                                   SQR(particles.ux3(p)));

          particles.i1(p)  = ctx.ni1 - 2;
          particles.dx1(p) = ONE - metric.template transform<1, Idx::XYZ, Idx::U>(
                                     x_dummy,
                                     math::abs(particles.ux1(p))) *
                                     remaining_dt_inv_energy;
        }
      }
    };

    template <class DOM>
    auto CustomParticleUpdate(simtime_t /*time*/, spidx_t sp, DOM& domain) const
      -> CustomPrtlUpdate {
      const auto m = domain.species[sp - 1].mass(); // sp is 1-indexed
      return CustomPrtlUpdate {
        domain.random_pool(),
        T_hot / m,
        T_cold / m,
        vd,
        global_xmin,
        global_xmax,
        bdry_accum
      };
    }

    /*
      Recalculate the cold-reservoir re-injection drift vd each step so the
      re-injected current cancels the outgoing current (Roberg-Clark 2018).
    */
    void CustomPostStep(timestep_t, simtime_t, Domain<S, M>&) {
      auto h_accum = Kokkos::create_mirror_view(bdry_accum);
      Kokkos::deep_copy(h_accum, bdry_accum);
      real_t s = h_accum(0); // sum of outgoing v_par at the cold reservoir
      real_t n = h_accum(1); // number of outgoing crossers
#if defined(MPI_ENABLED)
      real_t in[2] = { s, n }, out[2] = { ZERO, ZERO };
      MPI_Allreduce(in, out, 2, mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
      s = out[0];
      n = out[1];
#endif
      if (n > static_cast<real_t>(0.5)) {
        const real_t mean_out = s / n; // mean outgoing v_par (>0)
        const real_t vTc      = math::sqrt(TWO * T_cold);
        vd = invert_cold_mean(-mean_out, vTc);
      }
      Kokkos::deep_copy(bdry_accum, ZERO); // reset for next step
    }

    /*
      Custom field output: electron heat flux q_dir = sum_p (1/2) m v_dir v^2.
      Enable with output.fields.custom = ["Qx", "Qy", "Qz"]. Qx = q_|| along B0.
    */
    void CustomFieldOutput(const std::string&    name,
                           ndfield_t<M::Dim, 6>& buff,
                           std::size_t           idx,
                           timestep_t /*step*/,
                           simtime_t /*time*/,
                           const Domain<S, M>& domain) const {
      uint8_t dir = 1; // Qx by default
      if (name == "Qy" || name == "Q2") {
        dir = 2;
      } else if (name == "Qz" || name == "Q3") {
        dir = 3;
      }

      const auto use_weights  = params.template get<bool>("particles.use_weights");
      const auto inv_n0       = ONE / params.template get<real_t>("scales.n0");
      const auto smooth_order = params.template get<unsigned short>(
        "output.fields.smoothing.order");
      const auto smooth_method = OutputSmoothingType::from_string(
        params.template get<std::string>("output.fields.smoothing.method"));

      auto& electrons    = domain.species[0]; // heat flux is carried by the electrons
      auto  scatter_buff = Kokkos::Experimental::create_scatter_view(buff);
      Kokkos::parallel_for(
        "HeatFluxMoment",
        electrons.rangeActiveParticles(),
        HeatFluxMoment_kernel<S, M, 6>(dir,
                                       scatter_buff,
                                       static_cast<idx_t>(idx),
                                       electrons,
                                       use_weights,
                                       domain.mesh.metric,
                                       inv_n0,
                                       static_cast<uint8_t>(smooth_order),
                                       smooth_method));
      Kokkos::Experimental::contribute(buff, scatter_buff);
    }
  };
} // namespace user

#undef i_di_to_Xi
#endif
