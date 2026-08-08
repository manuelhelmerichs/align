# TODO -- symmetry-class completeness

## 1. Add the exact FRN `eps` sign symmetry

The ResNet-7 graph currently leaves six scalar `core/FRN_{0...5}/eps` leaves
unbound. The sampled FRN implementation evaluates `abs(eps)`, so each leaf has
an independent exact

$$
\varepsilon_j \mapsto -\varepsilon_j
$$

symmetry. Under the campaign's centered normal prior this also preserves the
prior and hence the posterior density. This is a genuine omission from the
exact graph, not merely a broader predictive equivalence.

Implementation:

- represent each FRN scalar as an independent size-one signed group, or add an
  equally explicit typed scalar-sign action if size-one
  `signed_permutation` groups make solver/component semantics awkward;
- bind only the corresponding `eps` leaf and never tie its sign to the FRN
  channel group;
- expose the deterministic canonical representative `eps >= 0`;
- keep the finite-epsilon restriction on pre-normalization kernel scaling:
  the `eps` sign symmetry does not make conv-kernel rescaling exact.

Required tests:

- independent sign flips preserve raw ResNet logits exactly;
- canonicalization is idempotent and maps negative `eps` to its positive
  representative;
- composition, inverse, serialization, component extraction, and random
  transform generation handle size-one sign groups;
- the ResNet coverage report moves from 428,256/428,272 to
  428,262/428,272, leaving only the ten class-bias coordinates outside the
  exact-logit graph.

## 2. Enlarge the GPT attention-circuit action

This is the scientifically important extension. Within an ordinary attention
head, the factorization admits

$$
Q \mapsto QA,\qquad K \mapsto KA^{-T}
$$

for invertible `A`, with an analogous contragredient transformation of the
value/output pair. The current LayerNorm-MHA recipe models positive diagonal
balancing and signed permutations, not the full orthogonal or general-linear
attention-circuit symmetry.

Coverage will remain 100% after this change; the orbit dimension and removable
geometry can nevertheless change substantially. Treat this as evidence that
coverage and symmetry-class completeness are different measurements.

Implement in stages:

1. **Orthogonal circuit action.** Add the orthogonal subgroup first and use
   Procrustes updates. Reuse the typed `MHACircuitConstraint`; do not infer the
   contragredient Q/K and V/O relationships from axis incidence.
2. **General invertible action.** Introduce a typed non-orthogonal transform
   whose application is role/circuit aware. The current generic graph action
   applies the same matrix on incident axes because it assumes orthogonality;
   simply adding `"general_linear"` to `TRANSFORM_FAMILIES` would be wrong.
3. **Gauge/canonical form.** Specify a well-conditioned representative for the
   non-compact GL orbit, including rank-deficient and repeated-singular-value
   behavior. Do not use an unconstrained least-squares match that can approach
   singular matrices merely to fit noise.

Required validation:

- end-to-end raw-logit invariance for random orthogonal and well-conditioned
  GL transforms;
- exact orbit recovery on planted real-weight copies;
- composition/inverse and artifact round trips;
- equivariance and idempotence of any new canonical form;
- explicit conditioning/rank failure gates;
- repeat the GPT independence-null calibration with the identical enlarged
  action on real and null ensembles before changing the correspondence claim.

Report raw gain, null gain, null excess, function-space barriers, runtime, and
conditioning diagnostics. A larger raw matched-vs-random gain alone is not
evidence of greater identifiability.
