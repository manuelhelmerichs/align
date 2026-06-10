"""L2 weight distance cost."""

import jax.numpy as jnp

from .base import CostFunction, register_cost_function


@register_cost_function("l2_weight")
class L2WeightCost(CostFunction):
    """L2 distance between reference layers and softly permuted targets."""

    name = "l2_weight"

    def compute_cost(
        self,
        spec,
        ref_views,
        target_views,
        soft_perms,
    ) -> jnp.ndarray:
        return self.compute_batched_cost(
            spec,
            ref_views,
            {
                gid: [jnp.expand_dims(v, 0) for v in target_views[gid]]
                for gid in target_views
            },
            {gid: jnp.expand_dims(perm, 0) for gid, perm in soft_perms.items()},
        )[0]

    def compute_batched_cost(
        self,
        spec,
        ref_views,
        target_views_batched,
        soft_perms,
    ) -> jnp.ndarray:
        loss = None
        for group_id, ref_list in ref_views.items():
            tgt_list = target_views_batched[group_id]
            perm = soft_perms[group_id]
            if perm.ndim == 2:
                perm = jnp.expand_dims(perm, 0)
            group_loss = jnp.zeros(
                (perm.shape[0],), dtype=jnp.asarray(ref_list[0]).dtype
            )
            for ref_view, tgt_view in zip(ref_list, tgt_list, strict=True):
                ref_mat = jnp.asarray(ref_view)
                tgt_mat = jnp.asarray(tgt_view)
                if tgt_mat.ndim == 2:
                    tgt_mat = jnp.expand_dims(tgt_mat, 0)
                permuted = jnp.einsum("bij,bjk->bik", perm, tgt_mat)
                diff = ref_mat[None, :, :] - permuted
                group_loss = group_loss + jnp.sum(jnp.square(diff), axis=(1, 2))
            loss = group_loss if loss is None else loss + group_loss
        if loss is None:
            return jnp.array([])
        return loss
