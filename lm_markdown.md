DEFINITIONS  (all evaluated at the current point θ)

  <a|b>   = 4 Re Σ_α ∫ a~*_α(f) b~_α(f) / S_α(f) df        α ∈ {A,E},  f ∈ (1e-5, f_Nyq)

  r(θ)    = s − h(θ)                                       residual
  Γ_ij(θ) = < ∂_i h | ∂_j h >                              Fisher matrix
  g_i(θ)  = < ∂_i h | r(θ) >                               CV gradient
  χ²(θ)   = < r | r >
  O(θ)    = <s|h> / sqrt( <s|s> <h|h> )                    overlap
  σ_i(θ)  = sqrt( |(Γ⁻¹)_ii| )                             Fisher 1σ

  ∂_i h from StableEMRIFisher (deriv_type="stable"), stencil refreshed every 5 iterations.


DAMPED CV STEP

  D_ij    = δ_ij · ( |Γ_ii| + 1e-30 )                      Marquardt scaling (diag of Γ)

  δ(λ)    = ( Γ + λ D )⁻¹ g

      λ → 0   :  δ = Γ⁻¹ g                  (bare Cutler–Vallisneri / Gauss–Newton step)
      λ → ∞   :  δ ≈ (1/λ) D⁻¹ g            (short scaled steepest-descent step)


λ UPDATE

  Quadratic model:   χ²(θ+δ) ≈ χ²(θ) − 2 δᵀg + δᵀΓδ

  Since (Γ + λD) δ = g   ⇒   Γδ = g − λDδ, the predicted decrease is

      L(δ) = 2 δᵀg − δᵀΓδ  =  δᵀ ( g + λ D δ )   > 0

  Actual decrease:   A(δ) = χ²(θ) − χ²(θ+δ)

  Ratio:             ρ = A(δ) / L(δ)


  ACCEPT  (ρ > 0):   θ ← θ + δ
                     λ ← λ · max( 1/3 ,  1 − (2ρ − 1)³ )
                     ν ← 2

        ρ = 1    ⇒  λ → λ/3        (model perfect: move toward pure CV)
        ρ = 1/2  ⇒  λ → λ          (unchanged)
        ρ → 0⁺   ⇒  λ → 2λ         (barely helped: damp harder)

  REJECT  (ρ ≤ 0, or Γ+λD singular):   θ unchanged, retry same iteration with
                     λ ← λ · ν
                     ν ← 2ν

        after k consecutive rejects from (λ₀, ν=2):
                     λ_k = λ₀ · 2^( k(k+1)/2 )          → ×2, ×8, ×64, ×1024, ...

  Up to 30 retries per iteration; if no λ gives ρ > 0, the climb stops.


TERMINATION   (any one)

  O(θ) > 1 − 1e-10
  A(δ) / χ²(θ) < 1e-9
  iteration = 150

  λ₀ = 1e-2 (house default; 1e-1 in notebook cell 11),  ν₀ = 2


FLAT-RIDGE CASE

  If g_i ≈ 0 for i ∈ {C_p, C_e}, then for every λ

      δ_i = Σ_j [ (Γ + λD)⁻¹ ]_ij g_j ≈ 0

  in those components, so no λ moves the fit off C_p = C_e = 0. That is why a
  stall gets the Nelder–Mead escape, not a change in λ.
