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

// helper: reconstruct a continuous cell coordinate Xi from (cell index, displacement)
#define i_di_to_Xi(I, DI) (static_cast<real_t>((I)) + static_cast<real_t>((DI)))

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    /*
      Background magnetic field. Btheta/Bphi are the field polar/azimuthal angles.
      The returned unit vector is the field in units of B0 (= scales.B0 = 1/larmor0);
      Entity normalizes the EM fields to B0, so magnitude 1 here = the fiducial B0.
    */
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

  // n0(x) propto 1/T0(x): density rises from 1 (hot, left) to temperature_gradient
  // (cold, right) so the electron pressure n*T stays uniform (Komarov et al. 2018).
  template <Dimension D>
  struct DensityGradient {
    DensityGradient(real_t reservoir_width, real_t x_max, real_t temperature_gradient)
      : reservoir_width { reservoir_width }
      , x_max { x_max }
      , temperature_gradient { temperature_gradient } {}

    Inline auto operator()(const coord_t<D>& x_Ph) const -> real_t {
      if (x_Ph[0] >= x_max - reservoir_width) { // cold reservoir at the right boundary
        return ONE * temperature_gradient;
      } else if (x_Ph[0] <= reservoir_width) { // hot reservoir at the left boundary
        return ONE;
      } else { // linear density gradient in the middle region
        return ONE + (temperature_gradient - 1) * (x_Ph[0] - reservoir_width) /
                       (x_max - 2 * reservoir_width);
      }
    }

  private:
    const real_t reservoir_width, x_max, temperature_gradient;
  };

  // initial T0(x): hot (temp*temperature_gradient) on the left, decreasing linearly
  // to cold (temp) on the right.
  template <Dimension D>
  struct MaxwellGradient {
    MaxwellGradient(random_number_pool_t& pool,
                    real_t                reservoir_width,
                    real_t                x_max,
                    real_t                temp,
                    real_t                temperature_gradient)
      : pool { pool }
      , reservoir_width { reservoir_width }
      , x_max { x_max }
      , temp { temp }
      , temperature_gradient { temperature_gradient } {}

    Inline void operator()(const coord_t<D>& x_Ph, vec_t<Dim::_3D>& v) const {
      auto T = temp;
      if (x_Ph[0] <= reservoir_width) {
        T = temp * temperature_gradient;
      } else if (x_Ph[0] >= x_max - reservoir_width) {
        T = temp;
      } else {
        T = temp * temperature_gradient -
            (temp * (temperature_gradient - 1) / (x_max - 2 * reservoir_width)) *
              (x_Ph[0] - reservoir_width);
      }
      arch::energy_dist::JuttnerSinge(v, T, pool);
    }

  private:
    random_number_pool_t pool;
    real_t               reservoir_width, x_max, temp, temperature_gradient;
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
      ::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D> {}
    };

    const SimulationParams& params;

    // domain properties
    const real_t global_xmin, global_xmax, reservoir_width;
    // gas properties
    const real_t temperature, temperature_gradient;
    // magnetic field properties
    const real_t  Btheta, Bphi;
    InitFields<D> init_flds;

    PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : params { p }
      , global_xmin { m.mesh().extent(in::x1).first }
      , global_xmax { m.mesh().extent(in::x1).second }
      , reservoir_width { p.template get<real_t>("setup.reservoir_width") }
      , temperature { p.template get<real_t>("setup.temperature") }
      , temperature_gradient { p.template get<real_t>("setup.temperature_gradient") }
      , Btheta { p.template get<real_t>("setup.Btheta", ZERO) }
      , Bphi { p.template get<real_t>("setup.Bphi", ZERO) }
      , init_flds { Btheta, Bphi } {}

    auto MatchFields(simtime_t) const -> InitFields<D> {
      return init_flds;
    }

    inline void InitPrtls(Domain<S, M>& domain) {
      const auto T_e = temperature / domain.species[0].mass();
      const auto maxwellian_e = MaxwellGradient<M::Dim>(domain.random_pool(),
                                                        reservoir_width,
                                                        global_xmax,
                                                        T_e,
                                                        temperature_gradient);
      const auto T_p = temperature / domain.species[1].mass();
      const auto maxwellian_p = MaxwellGradient<M::Dim>(domain.random_pool(),
                                                        reservoir_width,
                                                        global_xmax,
                                                        T_p,
                                                        temperature_gradient);
      const auto density_step = DensityGradient<M::Dim>(reservoir_width,
                                                        global_xmax,
                                                        temperature_gradient);

      arch::InjectNonUniform<S, M, decltype(maxwellian_e), decltype(maxwellian_p), decltype(density_step)>(
        params,
        domain,
        { 1, 2 },
        { maxwellian_e, maxwellian_p },
        density_step,
        ONE);

      domain.species[1].set_npart(0);
    }

    /*
      Reflecting thermal walls at the two x-boundaries (Komarov et al. 2018).
      A particle leaving through x=xmin (hot) / x=xmax (cold) is reflected back in
      with a fresh speed from the flux-weighted Maxwellian f0(u)*|v_x| at the wall
      temperature, advanced by the time-of-flight remaining after the collision.
      Runs inside the pusher BEFORE boundaryConditions and repositions the particle
      to i1=0 / ni1-2 (inside the domain), so the global x particle BC should be
      REFLECT, not PERIODIC: REFLECT will not double-reflect these already-inside
      particles, but it is a safe backstop that specularly bounces any crosser this
      kernel misses, instead of PERIODIC wrapping it from the hot wall to the cold
      reservoir (a silent gradient-corrupting teleport).
    */
    struct CustomPrtlUpdate {
      random_number_pool_t pool;
      real_t               temp_xmin; // hot wall (left)
      real_t               temp_xmax; // cold wall (right)
      real_t               xmin, xmax;

      CustomPrtlUpdate(random_number_pool_t& pool,
                       real_t                temp_xmin,
                       real_t                temp_xmax,
                       real_t                xmin,
                       real_t                xmax)
        : pool { pool }
        , temp_xmin { temp_xmin }
        , temp_xmax { temp_xmax }
        , xmin { xmin }
        , xmax { xmax } {}

      // Sample momentum from the flux-weighted Maxwellian f0(u)*|v_x| at temperature
      // T: draw from JuttnerSinge, accept with probability |beta_x|=|u_x|/gamma.
      static Inline void DrawFluxWeighted(vec_t<Dim::_3D>&            v,
                                          real_t                      T,
                                          const random_number_pool_t& pool) {
        constexpr int max_iter = 32;
        for (int it = 0; it < max_iter; ++it) {
          arch::energy_dist::JuttnerSinge(v, T, pool);
          const real_t g      = math::sqrt(ONE + SQR(v[0]) + SQR(v[1]) + SQR(v[2]));
          const real_t beta_x = math::abs(v[0]) / g;
          auto         rand_gen = pool.get_state();
          const real_t xi       = Random<real_t>(rand_gen);
          pool.free_state(rand_gen);
          if (xi < beta_x) {
            break;
          }
        }
      }

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

        // hot reflecting wall at the left boundary
        if (x_Ph < xmin) {
          DrawFluxWeighted(v, temp_xmin, pool);

          const int      delta_i1_to_wall  = particles.i1_prev(p);
          const prtldx_t delta_dx1_to_wall = particles.dx1_prev(p);
          const real_t   dx_to_wall = i_di_to_Xi(delta_i1_to_wall, delta_dx1_to_wall);
          const real_t   dt_to_wall = dx_to_wall /
                                    metric.template transform<1, Idx::XYZ, Idx::U>(
                                      x_dummy,
                                      beta_x_p);

          particles.ux1(p) = math::abs(v[0]);
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
          // cold reflecting wall at the right boundary
          DrawFluxWeighted(v, temp_xmax, pool);

          const int      delta_i1_to_wall  = ctx.ni1 - 1 - particles.i1_prev(p);
          const prtldx_t delta_dx1_to_wall = ONE - particles.dx1_prev(p);
          const real_t   dx_to_wall = i_di_to_Xi(delta_i1_to_wall, delta_dx1_to_wall);
          const real_t   dt_to_wall = dx_to_wall /
                                    metric.template transform<1, Idx::XYZ, Idx::U>(
                                      x_dummy,
                                      beta_x_p);

          particles.ux1(p) = -math::abs(v[0]);
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
      return CustomPrtlUpdate{domain.random_pool(),
                              temperature * temperature_gradient / m, // hot wall at xmin (left)
                              temperature / m,                        // cold wall at xmax (right)
                              global_xmin, 
                              global_xmax};
    }

    /*
      Custom field output: electron heat flux q_dir = sum_p (1/2) m v_dir v^2.
      Enable with output.fields.custom = ["Qx", "Qy", "Qz"]. Qx is the parallel
      heat flux (along B0/the gradient); the conduction-suppression result.
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
