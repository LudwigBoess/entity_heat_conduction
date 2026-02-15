#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/spatial_dist.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"

#include "kernels/particle_moments.hpp"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {

    /*
      Sets up background magnetic field for the simulation.

      @param btheta: magnetic field polar angle
      @param bphi: magnetic field azimuthal angle
    */
    InitFields(real_t btheta, real_t bphi)
      : Btheta { btheta * static_cast<real_t>(convert::deg2rad) }
      , Bphi { bphi * static_cast<real_t>(convert::deg2rad) } {}

    // magnetic field components
    Inline auto bx1(const coord_t<D>&) const -> real_t {
      return math::cos(Btheta);
    }

    Inline auto bx2(const coord_t<D>&) const -> real_t {
      return math::sin(Btheta) * math::sin(Bphi);
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return math::sin(Btheta) * math::cos(Bphi);
    }

  private:
    const real_t Btheta, Bphi;
  };

  template <SimEngine::type S, class M>
  struct DensityGradient : public arch::SpatialDistribution<S, M> {
    DensityGradient(const M& metric, real_t reservoir_width, real_t x_max, real_t temperature_gradient)
      : arch::SpatialDistribution<S, M> { metric }
      , reservoir_width { reservoir_width }
      , x_max { x_max } 
      , temperature_gradient { temperature_gradient } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
      if (x_Ph[0] > x_max - reservoir_width) { // cold reservoir at the right boundary
        return ONE * temperature_gradient;
      } else if (x_Ph[0] < reservoir_width) { // hot reservoir at the left boundary
        return ONE;
      } else { // linear density gradient in the middle region
        return ONE + (temperature_gradient - 1) * (x_Ph[0] - reservoir_width) / (x_max - 2 * reservoir_width);
      }
    }

  private:
    const real_t reservoir_width, x_max, temperature_gradient;
  };

  template <SimEngine::type S, class M>
  struct MaxwellGradient : public arch::EnergyDistribution<S, M> {
    
    MaxwellGradient(const M& metric, random_number_pool_t& pool, real_t reservoir_width, real_t x_max, real_t temp, real_t temperature_gradient) 
        : arch::EnergyDistribution<S, M>{metric}
        , pool {pool} 
        , reservoir_width { reservoir_width }
        , x_max { x_max } 
        , temp { temp } 
        , temperature_gradient { temperature_gradient }  {}

    Inline void operator()(const coord_t<M::Dim>& x_Ph, vec_t<Dim::_3D>& v) const {
      auto T = temp;
      // if the particle is in the hot reservoir, set its temperature to the hot temperature
      if (x_Ph[0] < reservoir_width) {
        T = temp * temperature_gradient;
      } else if (x_Ph[0] > x_max - reservoir_width) {
        // if the particle is in the cold reservoir, set its temperature to the cold temperature
        T = temp;
      } else {
        // if the particle is in the middle region, set its temperature according to a linear gradient
        T = temp * (1 + (temperature_gradient - 1) * (x_Ph[0] - reservoir_width) / (x_max - 2 * reservoir_width));
      }
      arch::JuttnerSinge(v, T, pool);
    }

  private:
    random_number_pool_t pool;
    real_t reservoir_width, x_max, temp, temperature_gradient;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics { traits::compatible_with<Metric::Minkowski>::value };
    static constexpr auto dimensions { traits::compatible_with<Dim::_2D, Dim::_3D>::value};

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    Metadomain<S, M>& global_domain;

    // domain properties
    const real_t    global_xmin, global_xmax, reservoir_width;
    // gas properties
    const real_t    temperature, temperature_gradient;
    // magnetic field properties
    real_t          Btheta, Bphi;
    InitFields<D>   init_flds;

    inline PGen(const SimulationParams& p, Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , global_domain { global_domain }
      , global_xmin { global_domain.mesh().extent(in::x1).first }
      , global_xmax { global_domain.mesh().extent(in::x1).second }
      , temperature { p.template get<real_t>("setup.temperature") }
      , temperature_gradient { p.template get<real_t>("setup.temperature_gradient") }
      , reservoir_width { p.template get<real_t>("setup.reservoir_width") }
      , Btheta { p.template get<real_t>("setup.Btheta", ZERO) }
      , Bphi { p.template get<real_t>("setup.Bphi", ZERO) }
      , init_flds { Btheta, Bphi }
      {}

    inline PGen() {}

    auto MatchFields(real_t time) const -> InitFields<D> {
      return init_flds;
    }

    inline void InitPrtls(Domain<S, M>& domain) {

      // define cold maxwellian
      const auto T_e = temperature / domain.species[0].mass();
      const auto maxwellian_e = MaxwellGradient<S, M>( domain.mesh.metric, domain.random_pool(), 
                                                  reservoir_width, global_xmax, T_e, temperature_gradient);

      const auto T_p = temperature / domain.species[1].mass();
      const auto maxwellian_p = MaxwellGradient<S, M>( domain.mesh.metric, domain.random_pool(), 
                                                  reservoir_width, global_xmax, T_p, temperature_gradient);

      // define density step
      const auto density_step = DensityGradient<S, M>(domain.mesh.metric, reservoir_width, global_xmax, temperature_gradient);

      // inject particles with a density step and a maxwellian energy distribution
      arch::InjectNonUniform<S, M, decltype(maxwellian_e), decltype(maxwellian_p), decltype(density_step)>(
        params,
        domain,
        { 1, 2 },
        { maxwellian_e, maxwellian_p },
        density_step,
        ONE);

    }

    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {

              // reset particle velocities in the reservoirs
      const auto left_threshold  = global_xmin + reservoir_width;
      const auto right_threshold = global_xmax - reservoir_width;
      // same maxwell distribution as above
      const auto mass_1   = domain.species[0].mass();
      const auto mass_2   = domain.species[1].mass();
      const auto t_hot_e  = temperature * temperature_gradient / mass_1;
      const auto t_hot_p  = temperature * temperature_gradient / mass_2;
      const auto t_cold_e = temperature / mass_1;
      const auto t_cold_p = temperature / mass_2;

      const auto maxwellian_hot_e = arch::Maxwellian<S, M>(
        domain.mesh.metric, domain.random_pool(),
        t_hot_e, { ZERO, ZERO, ZERO });
      const auto maxwellian_hot_p = arch::Maxwellian<S, M>(
        domain.mesh.metric,
        domain.random_pool(),
        t_hot_p,
        { ZERO, ZERO, ZERO });
      
      const auto maxwellian_cold_e = arch::Maxwellian<S, M>(
        domain.mesh.metric,
        domain.random_pool(),
        t_cold_e,
        { ZERO, ZERO, ZERO });
      const auto maxwellian_cold_p = arch::Maxwellian<S, M>(
        domain.mesh.metric,
        domain.random_pool(),
        t_cold_p,
        { ZERO, ZERO, ZERO });

      const auto& mesh    = domain.mesh;

      for (auto& s : { 1u, 2u }) {
        auto& species = domain.species[s - 1];
        auto  i1      = species.i1;
        auto  dx1     = species.dx1;
        auto  ux1     = species.ux1;
        auto  ux2     = species.ux2;
        auto  ux3     = species.ux3;
        auto  tag     = species.tag;

        Kokkos::parallel_for(
          "ResetParticles",
          species.rangeActiveParticles(),
          Lambda(index_t p) {
            if (tag(p) == ParticleTag::dead) {
              return;
            }
            const auto x_Cd = static_cast<real_t>(i1(p)) +
                              static_cast<real_t>(dx1(p));
            const auto x_Ph = mesh.metric.template convert<1, Crd::Cd, Crd::XYZ>(
              x_Cd);

            const coord_t<M::Dim> x_dummy { ZERO };

            // cold reservoir at the left boundary
            if (x_Ph < left_threshold) {
              vec_t<Dim::_3D> v_T { ZERO }, v_Cd { ZERO };
              if (s == 1u) {
                maxwellian_hot_e(x_dummy, v_T);
              } else {
                maxwellian_hot_p(x_dummy, v_T);
              }
              mesh.metric.template transform_xyz<Idx::T, Idx::XYZ>(x_dummy, v_T, v_Cd);
              ux1(p) = v_Cd[0];
              ux2(p) = v_Cd[1];
              ux3(p) = v_Cd[2];
            } else if (x_Ph > right_threshold) {
              vec_t<Dim::_3D> v_T { ZERO }, v_Cd { ZERO };
              if (s == 1u) {
                maxwellian_cold_e(x_dummy, v_T);
              } else {
                maxwellian_cold_p(x_dummy, v_T);
              }
              mesh.metric.template transform_xyz<Idx::T, Idx::XYZ>(x_dummy, v_T, v_Cd);
              ux1(p) = v_Cd[0];
              ux2(p) = v_Cd[1];
              ux3(p) = v_Cd[2];
            }
          });
      }
    }
  };
} // namespace user
#endif
