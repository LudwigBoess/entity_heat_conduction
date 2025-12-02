#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/field_setter.h"
#include "archetypes/problem_generator.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"

#include <algorithm>
#include <utility>

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    /*
      Set up constant magnetic field in x-direction
    */
    InitFields() {}

    // magnetic field components
    Inline auto bx1(const coord_t<D>&) const -> real_t {
      return ONE;
    }
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics { traits::compatible_with<Metric::Minkowski>::value };
    static constexpr auto dimensions {
      traits::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value
    };

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
    InitFields<D>   init_flds;

    inline PGen(const SimulationParams& p, Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , global_domain { global_domain }
      , global_xmin { global_domain.mesh().extent(in::x1).first }
      , global_xmax { global_domain.mesh().extent(in::x1).second }
      , temperature { p.template get<real_t>("setup.temperature") }
      , temperature_gradient { p.template get<real_t>("setup.temperature_gradient") }
      , reservoir_width { p.template get<real_t>("setup.reservoir_width") }
      , init_flds { } 
      {}

    inline PGen() {}

    auto MatchFields(real_t time) const -> InitFields<D> {
      return init_flds;
    }

    inline void InitPrtls(Domain<S, M>& domain) {

      // define temperatures of species
      const auto temperatures = std::make_pair(temperature,
                                               HALF * temperature);
      // inject particles
      arch::InjectUniformMaxwellians<S, M>(params, domain, ONE,
                                           temperatures, { 1, 2 });

      // set cold part of the initial distribution for electrons
      const auto& mesh    = domain.mesh;
      const auto box_half = (global_xmax - global_xmin) * HALF;

      // drift velocity to ensure zero net current
      const auto v_drift = ZERO; // todo

      // define cold maxwellian
      const auto t_cold_e = temperature / temperature_gradient / domain.species[0].mass();
      const auto maxwellian_cold_e = arch::Maxwellian<S, M>( domain.mesh.metric,
                                        domain.random_pool, t_cold_e, { v_drift, ZERO, ZERO });

        auto& species = domain.species[0];
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
            if (x_Ph > box_half) {
              vec_t<Dim::_3D> v_T { ZERO }, v_Cd { ZERO };
              // sample from cold maxwellian
              maxwellian_cold_e(x_dummy, v_T);
              mesh.metric.template transform_xyz<Idx::T, Idx::XYZ>(x_dummy, v_T, v_Cd);
              ux1(p) = v_Cd[0];
              ux2(p) = v_Cd[1];
              ux3(p) = v_Cd[2];
            }
          });
    }

    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {

              // reset particle velocities in the reservoirs
      const auto left_threshold  = global_xmin + reservoir_width;
      const auto right_threshold = global_xmax - reservoir_width;
      // same maxwell distribution as above
      const auto mass_1   = domain.species[0].mass();
      const auto mass_2   = domain.species[1].mass();
      const auto t_hot_e  = temperature / mass_1;
      const auto t_hot_p  = temperature / mass_2;
      const auto t_cold_e = temperature / temperature_gradient / mass_1;
      const auto t_cold_p = temperature / temperature_gradient / mass_2;

      const auto maxwellian_hot_e = arch::Maxwellian<S, M>(
        domain.mesh.metric, domain.random_pool,
        t_hot_e, { ZERO, ZERO, ZERO });
      const auto maxwellian_hot_p = arch::Maxwellian<S, M>(
        domain.mesh.metric,
        domain.random_pool,
        t_hot_p,
        { ZERO, ZERO, ZERO });
      
      const auto maxwellian_cold_e = arch::Maxwellian<S, M>(
        domain.mesh.metric,
        domain.random_pool,
        t_cold_e,
        { ZERO, ZERO, ZERO });
      const auto maxwellian_cold_p = arch::Maxwellian<S, M>(
        domain.mesh.metric,
        domain.random_pool,
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
                maxwellian_cold_e(x_dummy, v_T);
              } else {
                maxwellian_cold_p(x_dummy, v_T);
              }
              mesh.metric.template transform_xyz<Idx::T, Idx::XYZ>(x_dummy, v_T, v_Cd);
              ux1(p) = v_Cd[0];
              ux2(p) = v_Cd[1];
              ux3(p) = v_Cd[2];
            } else if (x_Ph > right_threshold) {
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
            }
          });
      }
    }
  };
} // namespace user
#endif
