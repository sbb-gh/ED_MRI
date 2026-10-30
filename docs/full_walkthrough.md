# Full workflow walkthrough

[README](../README.md) · [Getting started](toy_example.md)

## Typical usage pipeline

The typical EDMRI workflow comprises three stages:

1. **Prepare oversampled MRI data:** Define a comprehensive MRI acquisition protocol (the *superdesign*) and generate **paired input and target data** using either real MRI acquisitions or simulations. The input data contain MRI measurements acquired or simulated using the superdesign, while the target data represent the desired task outputs (e.g. quantitative parameter maps).

2. **Protocol optimisation:** Use TADRED to identify the subset of acquisitions that best supports the specified task (the optimised acquisition protocol) and train a task network that performs the task using only measurements from this protocol.

3. **Apply the optimised protocol and task network:** Acquire new real MRI data using the optimised acquisition protocol and apply the trained task network to generate the desired outputs. 

There are two approaches to protocol optimisation:

- **Real-data optimisation:** Acquire an oversampled real MRI dataset and use it to optimise the protocol.
- **Simulation-based optimisation:** Generate simulated MRI data and ground-truth labels to optimise the protocol without requiring an oversampled real MRI acquisition.

**Both approaches are intended to produce an optimised protocol and trained network that can be applied to real MRI data.** Simulated data can also be used during application for testing and benchmarking.

### 1. Prepare oversampled MRI data

#### Step 1.1: Define the superdesign

**Define an oversampled MRI acquisition protocol** that comprehensively samples the acquisition parameters of interest (e.g. b-values, gradient directions, echo times and diffusion times).

For real MRI, the superdesign should be a feasible but extensive research-grade acquisition, substantially longer than the intended optimised scan. By default, TADRED selects 25% of the acquisitions in the superdesign, so a useful rule of thumb is to make the superdesign four times as long as the desired optimised scan.

For simulations, the superdesign can be more extensive, provided the proportion of acquisitions selected by TADRED is adjusted accordingly. For example, if the superdesign contains ten times as many acquisitions as the desired optimised protocol, configure TADRED to select 10%.

**All acquisitions in the superdesign must be feasible on the intended scanner**, even when using simulations.

Save the superdesign as a text file, for example `superdesign.txt`, with:

- **Columns:** Acquisition parameters in a consistent order. For example, the first three columns could specify the gradient direction, with the fourth specifying the b-value, and any additional acquisition parameters in subsequent columns. Any column order can be used, provided it remains consistent throughout the workflow.
- **Rows:** Individual acquisitions. For example, the first row could specify a b = 0 s/mm² acquisition with gradient direction [0, 0, 0], and the second a b = 1000 s/mm² acquisition with gradient direction [1, 0, 0].

The resulting array should have dimensions **[T, M]**, where:

- `T` is the total number of acquisitions in the superdesign.
- `M` is the number of independent acquisition parameters.

#### Step 1.2: Generate input and target data

Choose one of the following approaches to generate the data used for optimisation.

**Option A: Optimisation using real MRI data**

1. **Acquire oversampled MRI data** using the superdesign. Save the data as a 4D NIfTI file, for example `superdesign_data.nii.gz`, with dimensions **[X, Y, Z, T]**.

2. **Generate the corresponding task outputs** using the complete oversampled dataset. For example, for parameter estimation, fit a model to the oversampled data to generate parameter maps. Save these as `superdesign_target.nii.gz`, with dimensions **[X, Y, Z, P]**, where `P` is the number of target parameters.

The oversampled MRI data serve as the **input data for TADRED**, while the corresponding task outputs serve as the **target data**.

**Option B: Optimisation using simulated MRI data**

1. **Generate ground-truth task labels** for `N` simulated voxels or samples. These should comprehensively cover the range of target values of interest. For example, for parameter estimation, generate combinations of ground-truth model parameters. Save these as `superdesign_target.npy`, with dimensions **[N, P]**, where `P` is the number of target parameters.

2. **Simulate MRI signals** using the superdesign and the generated ground-truth parameters. Save the simulated signals as `superdesign_data.npy`, with dimensions **[N, T]**.

The simulated MRI signals serve as the **input data for TADRED**, while the ground-truth labels serve as the **target data**.

This approach allows the acquisition protocol and task network to be optimised using simulations, before applying them to real MRI data.

**For both approaches:**

- Input and target data must correspond to the same voxels or samples.
- The number and order of acquisitions in the input data must match the rows of `superdesign.txt`.

### 2. Protocol optimisation

Once the superdesign, input data and target data have been prepared, run TADRED to optimise the acquisition protocol and train the task network.

For simulation-based optimisation:

```bash
edmri-optimise \
    data.input=superdesign_data.npy \
    data.target=superdesign_target.npy \
    data.superdesign=superdesign.txt \
    output.directory=optimisation_results
```

For real-data optimisation, replace the `.npy` filenames with the corresponding `.nii.gz` files.

This produces two main outputs:

1. **Optimised acquisition protocol:** A subset of acquisitions selected from the superdesign to efficiently perform the target task.

   Saved to:
   `optimisation_results/<UNIQUE_HASH>/tadred_results/optimised_protocol.txt`

	By default, the optimised protocol contains 25% of the acquisitions in the superdesign. This can be adjusted using the `optimisation.protocol_size` argument in `edmri-optimise`. For example, `optimisation.protocol_size=0.5` retains 50% of the acquisitions.

2. **Trained task network:** A neural network trained to perform the target task using only measurements from the optimised protocol.

   Saved to:
   `optimisation_results/<UNIQUE_HASH>/tadred_results/trained_task_network.pt`

The `<UNIQUE_HASH>` directory identifies the specific optimisation run and its configuration.

### 3. Apply the optimised protocol and task network

**The primary application of EDMRI is to use the optimised protocol and trained task network on new real MRI data**, regardless of whether optimisation was performed using real or simulated data.

#### Step 3.1: Acquire real MRI data

**Acquire new real MRI data** using `optimised_protocol.txt`.

Save the data as a 4D NIfTI file, for example `tadred_data.nii.gz`, with dimensions **[X, Y, Z, C]**, where `C` is the number of acquisitions in the optimised protocol.

The acquisitions must be performed in the same order as the optimised protocol used during training.

#### Step 3.2: Apply the trained task network

Run:

```bash
edmri-apply \
    task_network=optimisation_results/<UNIQUE_HASH>/tadred_results/trained_task_network.pt \
    data.input=tadred_data.nii.gz
```

The trained network performs the target task using the newly acquired MRI data. For example, for parameter estimation, it generates parameter maps from the optimised MRI measurements.

**Note:** `edmri-apply` does not require `optimised_protocol.txt` as an additional input because the task network has already been trained specifically for that protocol. However, the new MRI data must contain the correct acquisitions in the correct order.

#### Optional: Testing and benchmarking using simulated data

The optimised protocol and trained task network can also be evaluated using simulated MRI data.

1. **Simulate new MRI data** using `optimised_protocol.txt`. Save the data as `tadred_data.npy`, with dimensions **[N, C]**.

2. **Apply the trained task network:**

```bash
edmri-apply \
    task_network=optimisation_results/<UNIQUE_HASH>/tadred_results/trained_task_network.pt \
    data.input=tadred_data.npy
```

3. **Evaluate performance** by comparing the network predictions against the known ground-truth labels.

This provides a way to assess the accuracy and robustness of the optimised protocol before or alongside its application to real MRI data.
