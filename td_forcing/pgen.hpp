#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "traits/pgen.h"

#include "archetypes/energy_dist.h"
#include "archetypes/utils.h"
#include "framework/containers/particles.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

/*
  Thermodynamic-forcing setup of Pal Choudhury & Bott, "Modeling transport in
  weakly collisional plasmas using thermodynamic forcing" (arXiv:2504.14000v3).
  A uniform, periodic, magnetised electron plasma is driven by a velocity-dependent
  force that emulates a macroscopic temperature gradient of scale length L_T, so a
  parallel heat flux and the heat-flux-driven whistler instability develop without
  any spatial inhomogeneity, thermal wall or particle reservoir.

  Mapping the paper's Table 1 onto Entity's [scales] block, all lengths in code
  units:

    d_e    = skindepth0 / sqrt(n_e/n0),   n_e/n0 = setup.density / 2
    rho_e  = sqrt(2 * setup.temperature) * larmor0
    B0     = 1 / larmor0                        Table 1: B0 = sqrt(2 theta_e / beta_e)
    beta_e = 8 pi n_e T_e / B0^2 = (rho_e / d_e)^2

  d_e is skindepth0 and rho_e is larmor0 only for particular parameter choices:
  larmor0 is the gyroradius of a particle moving at c, and setup.density is the
  TOTAL density of the injected pair (see InitPrtls). Table 1 assumes d_e = 1, so
  it needs skindepth0 = 1 together with density = 2.
*/

namespace user
{
using namespace ntt;

/*
  Background static, spatially-uniform magnetic field (in units of B0 =
  1/larmor0). Btheta/Bphi are the polar/azimuthal angles; with the defaults
  (Btheta = 0) B points along x1, i.e. parallel to the default forcing/gradient
  direction, giving a parallel-heat-conduction setup.
*/
template <Dimension D> struct InitFields
{
    InitFields(real_t bmag, real_t btheta, real_t bphi)
        : Bmag{bmag}, Btheta{btheta * static_cast<real_t>(convert::deg2rad)},
          Bphi{bphi * static_cast<real_t>(convert::deg2rad)}
    {
    }

    Inline auto bx1(const coord_t<D> &) const -> real_t
    {
        return Bmag * math::cos(Btheta);
    }
    Inline auto bx2(const coord_t<D> &) const -> real_t
    {
        return Bmag * math::sin(Btheta) * math::sin(Bphi);
    }
    Inline auto bx3(const coord_t<D> &) const -> real_t
    {
        return Bmag * math::sin(Btheta) * math::cos(Bphi);
    }

  private:
    const real_t Bmag, Btheta, Bphi;
};

/*
  Temperature-gradient thermodynamic force (arXiv:2504.14000v3, Sec. III.1).

  A macroscopic electron temperature gradient of scale length L_T is emulated
  inside a homogeneous, periodic box by adding a spatially-uniform but
  velocity-dependent force to every particle. Eq. (54) writes it as an effective
  electric field

      E_eff = E + (1/2) (m_s v_th,s^2) / (q_s L_T) [ (gamma_p - 1)/theta_s - 3/2 ] a_hat,

  so the charge q_s cancels and the per-mass acceleration (Entity's external
  "force", du/dt) is, along the gradient direction a_hat,

      f_s = ( (gamma_p - 1) - (3/2) theta_s ) / L_T ,   theta_s = T / m_s .

  gamma_p is the individual particle's Lorentz factor, so the force accelerates
  particles above the mean energy along a_hat and decelerates the rest, skewing
  the distribution into a heat flux.

  Eq. (60) for the resulting drive on f carries -5/2 where this force carries
  -3/2. The extra -1 is the momentum-space divergence of the velocity-dependent
  force itself, d f_s / d p_par = p_par / (gamma_p L_T), not a second force term.

  The -(3/2) theta_s offset cancels the mean force only as theta_s -> 0, where
  <gamma_p - 1> -> (3/2) theta_s. At theta_e = 0.3 the sampled Maxwellian has
  <gamma_p - 1> = 0.356 against (3/2) theta_e = 0.45, leaving a residual mean
  force of -0.094 / L_T. It does not accumulate: a net electron drift is a
  current, and the uniform E_x that Ampere's law builds from it opposes the drift.

  Entity applies this through its external-force path (a particle-aware force
  setter fx*(x, particles, p)), folding it symmetrically into the Boris push the
  way the paper folds it into the Vay push; Appendix C tests the same force in a
  Boris pusher.
*/
template <Dimension D> struct TDForce
{
    TDForce(real_t theta_s, real_t inv_LT, int dir) : theta_s{theta_s}, inv_LT{inv_LT}, dir{dir}
    {
    }

    // per-mass acceleration magnitude for this particle
    Inline auto magnitude(const ntt::ParticleArrays &p, prtlidx_t idx) const -> real_t
    {
        const auto gamma = math::sqrt(ONE + SQR(p.ux1(idx)) + SQR(p.ux2(idx)) + SQR(p.ux3(idx)));
        return ((gamma - ONE) - static_cast<real_t>(1.5) * theta_s) * inv_LT;
    }

    Inline auto fx1(const coord_t<D> &, const ntt::ParticleArrays &p, prtlidx_t idx) const -> real_t
    {
        return (dir == 1) ? magnitude(p, idx) : ZERO;
    }
    Inline auto fx2(const coord_t<D> &, const ntt::ParticleArrays &p, prtlidx_t idx) const -> real_t
    {
        return (dir == 2) ? magnitude(p, idx) : ZERO;
    }
    Inline auto fx3(const coord_t<D> &, const ntt::ParticleArrays &p, prtlidx_t idx) const -> real_t
    {
        return (dir == 3) ? magnitude(p, idx) : ZERO;
    }

  private:
    const real_t theta_s, inv_LT;
    const int dir;
};

template <SimEngine::type S, class M> struct PGen
{
    static constexpr auto D{M::Dim};
    static constexpr auto engines{::traits::pgen::compatible_with<SimEngine::SRPIC>{}};
    static constexpr auto metrics{::traits::pgen::compatible_with<Metric::Minkowski>{}};
    static constexpr auto dimensions{::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>{}};

    const SimulationParams &params;

    // gas properties
    const real_t temperature; // T in units of m_e c^2 (theta_e for m_e = 1)
    const real_t density;     // TOTAL number density of the e-/p+ pair in units of n0
    // thermodynamic forcing
    const real_t L_T;       // temperature-gradient scale length [code units; Table 1 gives it in d_e]
    const int gradient_dir; // gradient/force direction a_hat: 1, 2 or 3
    // background magnetic field
    const real_t Bmag, Btheta, Bphi;
    InitFields<D> init_flds;

    PGen(const SimulationParams &p, const Metadomain<S, M> &)
        : params{p}, temperature{p.template get<real_t>("setup.temperature")},
          density{p.template get<real_t>("setup.density", ONE)}, L_T{p.template get<real_t>("setup.L_T")},
          gradient_dir{p.template get<int>("setup.gradient_dir", 1)}, Bmag{p.template get<real_t>("setup.Bmag", ONE)},
          Btheta{p.template get<real_t>("setup.Btheta", ZERO)}, Bphi{p.template get<real_t>("setup.Bphi", ZERO)},
          init_flds{Bmag, Btheta, Bphi}
    {
        // the force acts on the 4-velocity component ux{dir}; all three exist even
        // in 1D/2D, so dir in {1,2,3} is always valid regardless of D.
        raise::ErrorIf(gradient_dir < 1 or gradient_dir > 3, "setup.gradient_dir must be 1, 2 or 3", HERE);
    }

    /*
      Uniform plasma at rest (zero bulk drift), drawn from a Maxwellian at the
      given temperature. `density` is the TOTAL density of the pair: the injector
      puts ppc0 * density / 2 particles per cell into EACH species, so density = 2
      gives n_e = n0 and hence d_e = skindepth0, and N_ppc = ppc0 per species.

      The protons are then dropped. With pusher = "None" they never move and never
      deposit current, so they only contribute a static uniform charge density,
      which a periodic box with no Poisson solve does not see: the electrons behave
      exactly as if the neutralising background were still there, and the deleted
      species costs no memory or time.
    */
    inline void InitPrtls(Domain<S, M> &domain)
    {
        arch::InjectUniformMaxwellians<S, M>(params, domain, density, {temperature, temperature}, {1, 2});

        domain.species[1].set_npart(0);
    }

    // The thermodynamic force, applied to every species with its own
    // theta_s = T / m_s (sp is 1-indexed). Only the temperature-gradient component
    // of Eq. (53) is set up here; the bulk-velocity-gradient component, Eq. (62),
    // depends on the direction of the particle's motion and would need an
    // operator-split step after the push rather than an effective E field.
    auto ExternalFields(simtime_t /*time*/, spidx_t sp, const Domain<S, M> &domain) const -> std::pair<bool, TDForce<D>>
    {
        const auto m_s = domain.species[sp - 1].mass();
        return {true, TDForce<D>{temperature / m_s, ONE / L_T, gradient_dir}};
    }

    /*
      Resample runaway particles back onto a Maxwellian. Particles whose
      perpendicular energy puts them past the runaway condition of Eqs. (72) and
      (75) are accelerated without bound by the temperature-gradient force, which
      is outside its formal regime of validity (Sec. III.2 and V of the paper).

      Any particle with u^2 > 16 * u_lim^2 (u_lim = sqrt(2 * theta_s), i.e.
      |u| > 4 * u_lim) is redrawn from the same zero-drift Maxwellian used to
      initialize the plasma, with the species' own theta_s = T / m_s. At
      theta_e = 0.3 that threshold is 5.66 sigma_1D and holds 5.6e-7 of a thermal
      population, so it caps the runaway tail without touching the electrons that
      carry the heat flux.
    */
    void CustomPostStep(timestep_t /*step*/, simtime_t /*time*/, Domain<S, M> &domain)
    {
        const auto metric = domain.mesh.metric;
        for (std::size_t s = 0; s < domain.species.size(); ++s)
        {
            auto &species = domain.species[s];
            const auto theta_s = temperature / species.mass();
            const auto u_thr2 = static_cast<real_t>(16.0) * (TWO * theta_s); // (4 * u_lim)^2

            const auto maxwellian = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(domain.random_pool(), theta_s);

            auto ux1 = species.ux1;
            auto ux2 = species.ux2;
            auto ux3 = species.ux3;
            auto tag = species.tag;

            Kokkos::parallel_for(
                "TDForceResample", species.rangeActiveParticles(), Lambda(prtlidx_t p) {
                    if (tag(p) == ParticleTag::dead)
                    {
                        return;
                    }
                    const auto u2 = SQR(ux1(p)) + SQR(ux2(p)) + SQR(ux3(p));
                    if (u2 > u_thr2)
                    {
                        const coord_t<M::Dim> x_dummy{ZERO};
                        vec_t<Dim::_3D> v_T{ZERO}, v_Cd{ZERO};
                        maxwellian(x_dummy, v_T);
                        metric.template transform_xyz<Idx::T, Idx::XYZ>(x_dummy, v_T, v_Cd);
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
