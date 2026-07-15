Absolute Damage
===============

Introduction
------------

The absolute damage option allows model providers to include absolute damage amounts rather than damage factors in the
damage bin dictionary.

The damage bin dictionary supports the column `damage_type` which can be used to specify whether a given row corresponds
to a damage factor or absolute damage.

By default the `damage_type=0`, in this case if the damage factors are less than or equal to 1 in the damage bin dictionary, the factor will
be applied as normal during the loss calculation, by applying the sampled damage factor to the TIV to give a simulated
loss; but when the factor is greater than 1, the TIV is not used in the calculation at all, but rather the absolute
damage is applied as the loss.

**Example**

    **Example 1 (damage factor):** if the sampled damage factor is 0.6 and the TIV is 100,000, the sampled loss will be 60,000

    **Example 2 (absolute damage):** if the sampled damage factor is 500 and the TIV is 100,000, the sampled loss will be 500

A `damage_type=1` corresponds to the damage bin being treated as a damage factor.

A `damage_type=2` corresponds to the damage bin being treated as an absolute damage.
|
An example toy model with the absolute damage factor option is availible to use from `here <https://github.com/OasisLMF/
OasisModels/tree/develop/PiWindAbsoluteDamage>`_.

A more involved toy model incorporating the use of the `damage_type` in the damage bin dictionary in the context of
business interruption can seen `here <https://github.com/OasisLMF/OasisModels/tree/bi-test_model/PiWindBI>`.
