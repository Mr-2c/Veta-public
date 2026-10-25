# src/models — numpy design & verification models

These are **Python (numpy) models that back the Fortran solver** — not part of the
build. Each one validates algebra or quantifies a design decision
*before/independently of* the compiled code: the index/layout/precision logic is
modelled in numpy first. They live under `src/`
because they are tied to specific source modules, distinct from the user-facing
`tools/` (checkpoint/snapshot readers, stats analysis) and from `examples/`
(input decks).

| model | backs / answers |
|-------|-----------------|
| `m2_distributed_transform_model.py` | the 1-D slab transpose ordering (P-independent, bit-for-bit vs monolithic) |
| `work_decomp_model.py`              | slab decomposition counts/displacements |
| `work_m1_direct_a2a_model.py`       | the Layout-B staging identity + the direct per-(peer,field) NCCL exchange (bit-exact sweep) |
| `work_io_layout_model.py`           | checkpoint/snapshot global-order I/O layout |
| `work_pressure_model.py`            | the diagnostic Neumann-Poisson pressure solve |
| `work_pressure_rhs_model.py`        | the pressure RHS (∇·N) assembly |
| `work_galilean_model.py`            | the Galilean convecting frame (vorticity unchanged, mean forcing invariant) |

Run any of them directly with `python3 src/models/<name>.py`.
