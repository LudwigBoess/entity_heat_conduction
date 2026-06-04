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

    // electric field components
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

  template <SimEngine::type S, class M>
  struct ConstantDensity : public arch::SpatialDistribution<S, M> {
    ConstantDensity(const M& metric)
      : arch::SpatialDistribution<S, M> { metric } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
      return ONE;
    }
  };

  template <SimEngine::type S, class M>
  struct RC_edist : public arch::EnergyDistribution<S, M> {
    
    RC_edist(const M& metric, random_number_pool_t& pool, real_t T_hot, real_t T_cold, real_t T_perp, real_t T_par, 
                    real_t density_factor, real_t initial_drift_factor) 
        : arch::EnergyDistribution<S, M>{metric}
        , pool {pool} 
        , T_hot { T_hot }
        , T_cold { T_cold } 
        , T_perp { T_perp } 
        , T_par { T_par }  
        , density_factor { density_factor }
        , initial_drift_factor { initial_drift_factor } {}

    Inline void operator()(const coord_t<M::Dim>& x_Ph, vec_t<Dim::_3D>& v) const {
      
      auto rand_gen = pool.get_state();
      auto rnd_num = Random<real_t>(rand_gen);
      pool.free_state(rand_gen);

      // hot reservoir
      if (rnd_num >= (density_factor/ (1 + density_factor))) {
        arch::JuttnerSinge(v, T_hot, pool);
        v[1] = math::abs(v[1]);
      // cold reservoir
      } else {
        arch::JuttnerSinge(v, T_perp, pool);

        // save x and z components of velocity for later use in the parallel direction
        real_t dummy_x = v[0];
        real_t dummy_z = v[2];

        arch::JuttnerSinge(v, T_cold, pool);
        v[1] = v[1] - initial_drift_factor * math::sqrt(TWO * T_cold); // electron mass is always 1
        
        while (v[1] > ZERO) {
          arch::JuttnerSinge(v, T_perp, pool);
          dummy_x = v[0];
          dummy_z = v[2];
          arch::JuttnerSinge(v, T_cold, pool);
          v[1] = v[1] - initial_drift_factor * math::sqrt(TWO * T_cold);
        }
        v[0] = dummy_x;
        v[2] = dummy_z;
      }
    }

  private:
    random_number_pool_t pool;
    real_t T_hot, T_cold, T_perp, T_par;
    real_t density_factor, initial_drift_factor;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines {
      arch::traits::pgen::compatible_with<SimEngine::SRPIC>::value
    };
    static constexpr auto metrics {
      arch::traits::pgen::compatible_with<Metric::Minkowski>::value
    };
    static constexpr auto dimensions {
      arch::traits::pgen::compatible_with<Dim::_2D, Dim::_3D>::value
    };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    Metadomain<S, M>& global_domain;

    // gas properties
    const real_t    T_ions, T_hot, T_cold, T_perp, T_par;
    const real_t    density_factor, initial_drift_factor;

    // magnetic field properties
    real_t          Btheta, Bphi;
    InitFields<D>   init_flds;

    inline PGen(const SimulationParams& p, Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , global_domain { global_domain }
      , T_ions { p.template get<real_t>("setup.T_ions") }
      , T_hot { p.template get<real_t>("setup.T_hot") }
      , T_cold { p.template get<real_t>("setup.T_cold") }
      , T_perp { p.template get<real_t>("setup.T_perp") }
      , T_par { p.template get<real_t>("setup.T_par") }
      , density_factor { p.template get<real_t>("setup.density_factor") }
      , initial_drift_factor { p.template get<real_t>("setup.initial_drift_factor") }
      , Btheta { p.template get<real_t>("setup.Btheta", ZERO) }
      , Bphi { p.template get<real_t>("setup.Bphi", ZERO) }
      , init_flds { Btheta, Bphi }
      {}

    inline PGen() {}

    auto MatchFields(real_t time) const -> InitFields<D> {
      return init_flds;
    }

    inline void InitPrtls(Domain<S, M>& domain) {

      // define maxwellian of background ions
      const auto T_p = T_ions / domain.species[1].mass();
      const auto maxwellian_p = arch::Maxwellian<S, M>( domain.mesh.metric, domain.random_pool(), T_p);

      // define the electron distribution function with a hot and cold component, and a density step between them
      const auto electron_dist = RC_edist<S, M>( domain.mesh.metric, domain.random_pool(), 
                                                  T_hot, T_cold, T_perp, T_par, density_factor, initial_drift_factor);

      // define density step
      const auto density = ConstantDensity<S, M>(domain.mesh.metric);

      // inject particles with a density step and a maxwellian energy distribution
      arch::InjectNonUniform<S, M, decltype(electron_dist), decltype(maxwellian_p), decltype(density)>(
        params,
        domain,
        { 1, 2 },
        { electron_dist, maxwellian_p },
        density,
        ONE);

      //domain.species[1].set_npart(0); // remove protons to start with only electrons in the domain

    }


  };
} // namespace user
#endif
