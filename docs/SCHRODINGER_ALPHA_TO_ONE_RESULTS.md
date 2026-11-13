# One-pager: Schrödinger from the bare CAL at α → 1⁻

**Paper:** *Schrödinger evolution from the bare Causal Accumulation Law at α→1⁻: feedback on the accumulated past, unitarity in the limit, and carrier entanglement as time dilation*. PDF at `docs/paper/cal_schrodinger_collapse.pdf`, source in the `.tex` next to it.

**Reproduce:** `python3 scripts/cal_edge_transducer.py`. It needs numpy only and takes about 15 s.

## Inputs
- **The CAL as written:** T = αJ − (1−α)∫k ΠPΦ J.
- **The free, flat sector:** ΠPΦ = 1, an order-α power-law memory, and a spatial profile ĝ(k).
- **The phase belongs to the source.** An order-α accumulation turns the phase channel e^{iS/ħ} by α quarter-turns. No factor multiplies the memory.
- **The update rule:** each output becomes the next current, J_{n+1} = T_n. This is a feedback loop on the accumulated past.
- **Orientation:** the current's past rotated one way.

**Not used:** a phase prefactor, a coupling constant, a time unit, a double scaling, or T = 0 as a law. Time is counted in updates and quoted in radians of the carrier's own rotation.

## Results

| Result | Statement | Check |
|---|---|---|
| Rotation | Below 1 the past pulls back, and the current turns by √(1−α) per update. That rotation is the *i* of Schrödinger. | ratio 0.9897 → 1.0000 for 1−α = 10⁻² … 10⁻⁷ |
| Dispersion and mass | The rotation slows as √ĝ(k), so ħ/2m = ⟨r²⟩/12 in rest-rotation units, set by the kernel width. | ratio 1.0000 to 0.990 |
| Unitarity in the limit | For α < 1 the evolution is a contraction: exp[−(i + √(1−α))(ħ/2m)k²t − (π/4)(1−α)t]. It is unitary as α → 1⁻. | decay per radian ÷ (1−α): 0.901 → 0.787 (π/4 = 0.785); diffusion ratio → √(1−α) |
| Wavepacket | Converges to Schrödinger, down to the k⁴ floor, which halves with ⟨r²⟩. | error 1.46×10⁻² against a 1.26×10⁻² floor at 1−α = 10⁻⁴ |
| Orientation | A one-way past keeps one direction. The whole past is needed: a two-cycle history leaks 70–97%. | leakage ~10⁻⁸ |
| Born needs one orientation | With a one-way past, the closure and readout drift → 0. With a zero past, the readout drift stays at 3.7×10⁻². | Table 5 |
| Feedback across α = 1 | Frozen at 1. Above 1, runaway at √(α−1) (causal) or an advanced bounded response (ALD's two horns). Braking plus fading settles it; braking alone cannot (windup). | growth 0.101 vs 0.1; settles for b > 0.095 at λ = 0.1 |
| Normalizable memory | Every root lies strictly inside the unit circle, so nothing sustains a rotation. The power law escapes because its total weight is infinite. | proof + companion-matrix roots |
| Loss | Loss is contraction from rank 2 to rank 1. It is absent in the free edge, present in interaction and readout, and that is where the Landauer cost is paid. | — |
| Kept | Galilean edge; no interaction (one order-1 leg); one carrier = two factors with C = 1/γ; readout survives composition; Λ²J = 0. | readout checks at 10⁻¹⁵–10⁻¹⁶ |

## What changed from the earlier version
- The prefactor, the coupling r₁, the time unit τ_c, the double scaling, and T = 0 as a law are all removed.
- Unitarity was exact for every α < 1 only because the prefactor cancelled the memory's phase. Now it is exact only at the edge.
- The mass formula is now ħ/2m = ⟨r²⟩/12 in rest radians.
- The old script `scripts/cal_schrodinger_collapse.py` is retired. Its readout checks moved to `cal_edge_transducer.py` as section H.

## Open
- A rigorous remainder estimate for the asymptotic expansion. It is formal now, and confirmed numerically down to 1−α = 10⁻⁷.
- The equivalence condition (kernel width = √6 Compton wavelengths), as a forward pointer to the α = 2 / gravity program.
- Interaction, which needs the return leg (order 2).
