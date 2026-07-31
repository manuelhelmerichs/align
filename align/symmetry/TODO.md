# TODO -- symmetry-class completeness

The parameter-coverage report answers whether at least one modeled graph
binding touches a scalar coordinate. It does **not** establish that the
transform family acting on covered coordinates is maximal. This distinction is
load-bearing for the scale comparison: the current GPT graph covers 100% of
the stored parameter vector, but its attention-circuit action is much smaller
than the full exact factorization symmetry.

Keep three equivalence boundaries separate:

1. exact raw-output symmetries represented by `SymmetryGraph`;
2. predictive-distribution equivalences that may change logits, kept outside
   the exact graph;
3. function-preserving relations that are not group actions on one fixed-width
   parameter vector.

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

## 2. Do not fold softmax translation into the exact-logit graph

For a classifier head,

$$
W \mapsto W + v\mathbf 1^\top,\qquad
b \mapsto b + c\mathbf 1
$$

adds a common input-dependent scalar to every logit. Softmax probabilities and
the categorical likelihood are invariant, but raw logits are not. The
existing opt-in `center_softmax_head` stage is therefore the correct
abstraction boundary.

Follow-up:

- document an optional predictive-probability coverage view if it becomes
  useful, but do not combine it numerically with exact-logit graph coverage;
- retain raw-logit drift as the exact-symmetry certificate;
- do not claim the ten ResNet class-bias entries are missing exact graph
  symmetries merely because the probability-level translation touches them.

The MLP output bias and log observation scale have no analogous exact
symmetry: they change the regression distribution and should remain fixed.

## 3. Enlarge the GPT attention-circuit action

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

## 4. Keep semi-permutations and width-changing relations separate

Neuron splitting/merging and semi-permutations can relate different-width
networks, but they are not ordinary invertible group actions on a single fixed
parameter vector. Do not force them into `SymmetryGroup` or use parameter
coverage as their scope metric.

If pursued, introduce a separate typed correspondence/morphism abstraction
with explicit source and target shapes, then test function preservation and
null calibration independently. This comes after the fixed-width
orthogonal/GL attention work.

## Completion criteria

- The exact graph contains every implemented exact raw-output action and keeps
  probability-only or width-changing equivalences explicitly separate.
- Coverage reports state both coordinate coverage and transform family; no
  manuscript claim treats 100% coordinate coverage as maximality.
- The primary and many-basin artifacts, diagnostics results, paper overview,
  manuscript appendix, wiki architecture pages, and tests are updated together
  for any implemented extension.
