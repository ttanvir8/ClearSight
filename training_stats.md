# Training Speed Analysis

**Session Duration**: 11 hours 28 minutes (41,280 seconds)
**Total Steps**: 3,570
**Average Time Per Step**: **11.56 seconds**

## Breakdown
*   **Total Elapsed Time**: 41,280 seconds
    *   (11 hours × 3600) + (28 minutes × 60)
*   **Performance**:
    *   **Data Loading**: ~0.001s / step (Negligible)
    *   **Computation (GPU)**: ~11.56s / step
        *   This indicates the T4 GPUs are heavily utilized by the `DiffusionUNet` model with `batch_size=4` (effective 64 patches).
