![alt text](/_static/images/kernel/banner.jpg "banner")
# Appendix C: Multi-peril model support <a id="multiperil"></a>

ktools now supports multi-peril models through the introduction of the coverage_id in the data structures.  

Ground up losses apply at the “Item” level in the Kernel which corresponds to “interest coverage” in business terms, which is the element of financial loss that can be associated with a particular asset. In ktools, item_id represents the element of financial loss and coverage_id represents the asset with its associated total insured value. If there is more than one item per coverage (as defined in the items data) then each item represents an element of financial loss from a particular peril contributing to the total loss for the asset. For each item, the identification of the peril is held in the areaperil_id, which is a unique key representing a combination of the location (area) and peril.

#### Multi-peril damage calculation

Ground up losses are calculated by multiplying the damage ratio for an item by the total insured value of its associated coverage (defined in the coverages data).  The questions are then; how are these losses combined across items, and how are they correlated?

There are a few ways in which losses can be combined and the first example in ktools uses a simple rule, which is to sum the losses for each coverage and cap the overall loss to the total insured value. This is what you get when you use the -c parameter in gulcalc to output losses by 'coverage'.  

In v3.1.0 the method of combining losses became function-driven using the gulcalc command line parameter -a as a few standard approaches have emerged. These are;

| Allocation option | Description                                                                                           | 
|:------------------|:------------------------------------------------------------------------------------------------------|
| 0                 | Do nothing (suitable for single sub-peril models with one item per coverage)                          |
| 1                 | Sum damage ratios and cap to 1. Back-allocate in proportion to contributing subperil loss             |
| 2                 | Total damage =  maximum subperil damage. Back-allocate all to the maximum contributing subperil loss  |
| 3                 | Multiplicative method for combining damage. Back-allocate in proportion to contributing subperil loss |

Allocation options 0,1,2 have been implemented to date. 

Correlation of item damage is generic in ktools, as damage can either be 100% correlated or independent (see [Appendix A Random Numbers](RandomNumbers.md)). This is no different in the multi-peril case when items represent different elements of financial loss to the same asset, rather than different assets.  More sophisticated methods of multi-peril correlation have been implemented for particular models, but as yet no standard approach has been implemented in ktools.

Note that ground up losses by item can be passed into the financial module unallocated (allocation method 0) using the gulcalc option -i, or allocated using the gulcalc option -a1 -i.  If the item ground up losses are passed though unallocated then the limit of total insured value must be applied as part of the financial module calculations, to prevent the ground up loss exceeding the coverage TIV.  

[Return to top](#multiperil)

[Back to Contents](Contents.md)
