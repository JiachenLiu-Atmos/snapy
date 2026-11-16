# FP32 restart and coupled continuation (M5)

An anomaly checkpoint continues to contain physical `hydro_u` totals, exact
`hydro_u_stored` anomalies, the FP64 `hydro_ref`, and the original composition
in `hydro_ref_species`. Do not recover small anomalies by subtracting the
reference from rounded totals when the exact stored field is available.

New anomaly checkpoints add `hydro_restart_metadata`, an int64 vector
`[1, storage_bits]`, with storage_bits equal to 32 or 64. Version 1 covers the
existing reference representation and adds `hydro_ref_x1f`, `hydro_ref_x2f` and
`hydro_ref_x3f`, copied from the block's coordinate faces. These fields are
per-block, including inside the existing restart bundle. They are consumed by
the loader and are not returned as evolving simulation variables.

The loader checks the version, stored dtype/shape, required state/reference
fields and exact coordinate faces before installing a background. A different
grid requires explicit remapping of totals and initialization of a new
background; loading a same-shaped checkpoint on a changed grid is rejected.
This does not implement regridding. Existing geometry-cache invalidation still
applies to normal initialization and coordinate changes.

Files without this metadata remain readable using the previous format.
They cannot provide the new grid identity check. Perturbation fluxes require
the original density/energy and species backgrounds even for legacy files;
current composition must not be substituted for initial composition.

With perturbation fluxes enabled, the destination `dynamics/precision` selects
the stored hydro and scalar dtype. FP32-to-FP64 resume promotes exact stored
anomalies; FP64-to-FP32 resume rounds anomalies once. Geometry, backgrounds,
total primitives and characteristic-outflow references remain FP64. Same-dtype
continuation is exact with unchanged physics/configuration and arithmetic;
cross-dtype evolution is compared numerically. Legacy
anomaly-storage emulation without perturbation fluxes retains its saved dtype.

Replacing a mesh may leave old Python block references alive temporarily. Local
layout cleanup removes only its own registration, so releasing those references
does not unregister the replacement mesh.

The reduced Jupiter test reconstructs hydro, thermodynamic and kinetic modules
at a checkpoint. Chemistry continues through the existing FP64 kintera APIs,
applying only species increments to the stored state. No kintera change or new
cross-repo API is required. The test also checks scalar continuation.

Validation: `ctest --test-dir build --output-on-failure -L fp32-restart` covers
heated and balanced cases, characteristic outflow, both precision-switch
directions, legacy/corrupt metadata, grid mismatch, four-block bundles, and
exact moist/chemistry continuation in FP32 and FP64. Results are recorded in
[FP32_PROGRESS.md](../../FP32_PROGRESS.md). Production paddle drivers, external
Jupiter decks, distributed ranks and GPU execution require separate validation.
