#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/field_setter.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/spatial_dist.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"

#include "kernels/particle_moments.hpp"

#include <algorithm>
#include <utility>

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
  struct DensityGradient : public arch::SpatialDistribution<S, M> {
    DensityGradient(const M& metric, real_t reservoir_width, real_t x_max, real_t temperature_gradient)
      : arch::SpatialDistribution<S, M> { metric }
      , reservoir_width { reservoir_width }
      , x_max { x_max } 
      , temperature_gradient { temperature_gradient } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
      if (x_Ph[0] >= x_max - reservoir_width) { // cold reservoir at the right boundary
        return ONE * temperature_gradient;
      } else if (x_Ph[0] <= reservoir_width) { // hot reservoir at the left boundary
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
      if (x_Ph[0] <= reservoir_width) {
        T = temp * temperature_gradient;
      } else if (x_Ph[0] >= x_max - reservoir_width) {
        // if the particle is in the cold reservoir, set its temperature to the cold temperature
        T = temp;
      } else {
        // if the particle is in the middle region, set its temperature according to a linear gradient
        T = temp * temperature_gradient - (temp * (temperature_gradient - 1) / (x_max - 2 * reservoir_width)) * (x_Ph[0] - reservoir_width);
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
    static constexpr auto engines {
      arch::traits::pgen::compatible_with<SimEngine::SRPIC>::value
    };
    static constexpr auto metrics {
      arch::traits::pgen::compatible_with<Metric::Minkowski>::value
    };
    static constexpr auto dimensions {
      arch::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value
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

    struct CustomPrtlBC {
      random_number_pool_t pool;
      real_t temp_cold, temp_hot;

      template <class PusherKernel>
      Inline void operator()(index_t p, int dim, bool is_min, const PusherKernel& pusher) const {
        vec_t<Dim::_3D> v {ZERO};

        // Reflecting boundary that resamples velocity
        if (dim == 1) {
          if (is_min) {
            arch::JuttnerSinge(v, temp_cold, pool);

            pusher.i1(p)  = 0;
            pusher.dx1(p) = ONE - pusher.dx1(p);
            pusher.ux1(p) = -pusher.ux1(p);
            //pusher.ux1(p) = v[0];
            //pusher.ux2(p) = v[1];
            //pusher.ux3(p) = v[2];
          } else {
            arch::JuttnerSinge(v, temp_hot, pool);

            pusher.i1(p)  = pusher.ni1 - 1;
            pusher.dx1(p) = ONE - pusher.dx1(p);
            pusher.ux1(p) = -pusher.ux1(p);
            //pusher.ux1(p) = v[0];
            //pusher.ux2(p) = v[1];
            //pusher.ux3(p) = v[2];
          }
          
        }
      }
    };

    template <class D>
    auto CustomParticleBoundary(simtime_t /*time*/, spidx_t sp, D& domain) const {
      return CustomPrtlBC{
        domain.random_pool(),
        temperature / domain.species[sp].mass(), 
        temperature_gradient * temperature / domain.species[sp].mass()
      };
    }
  };
} // namespace user
#endif
