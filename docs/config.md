# Configuration files

EDMRI uses **YAML configuration files** to manage experiment settings. These files specify the input data, acquisition protocol, optimisation parameters, neural network training settings and output locations.

Configuration files make experiments easier to organise, reproduce and modify without changing the Python code. Default settings can also be overridden directly from the command line.

## Default configuration files

EDMRI provides two default configuration files, corresponding to its two main commands:

| Configuration file | Command | Purpose |
|---|---|---|
| `edmri/conf/optimise_default.yaml` | `edmri-optimise` | Configures acquisition protocol optimisation and task network training. |
| `edmri/conf/apply_default.yaml` | `edmri-apply` | Configures application of a trained task network to new MRI data. |

Each command loads its corresponding default configuration file unless an alternative is specified.

## Optimisation configuration

The `optimise_default.yaml` file controls the optimisation process. It contains four main sections:

- **`data`:** Input MRI data, corresponding task targets, superdesign acquisition protocol and optional mask.
- **`optimisation`:** Optimisation method, number of acquisitions to retain, number of optimisation iterations and random seed.
- **`training`:** Neural network architecture and training hyperparameters.
- **`output`:** Output directory and whether to force an existing optimisation run to be repeated.

### Annotated `optimise_default.yaml`

```yaml
# Default configuration for edmri-optimise.

# ============================================================
# Input data
# ============================================================

data:
  # Oversampled MRI measurements.
  # Supports simulated data (.npy) or real MRI data (.nii.gz).
  input: examples/data/sandi/superdesign_signals.npy

  # Corresponding task targets, e.g. ground-truth model
  # parameters or quantitative parameter maps.
  target: examples/data/sandi/superdesign_targets.npy

  # Acquisition protocol used to generate the input data.
  # Each row represents one acquisition.
  superdesign: examples/data/sandi/superdesign_protocol.txt

  # Optional mask for NIfTI data.
  # Set to null to use all voxels.
  mask: null

# ============================================================
# Acquisition optimisation
# ============================================================

optimisation:
  method: tadred 	# Optimisation method.

  protocol_size: 0.25	# Fraction of superdesign acquisitions to retain.
  
  n_iterations: 5	# Number of TADRED reduction stages.
  
  random_state: 42	# Random seed for reproducibility.

# ============================================================
# TADRED training
# ============================================================

training:
  network:    
    num_units_score: [1000, 1000]		# Hidden layer sizes of the subsampling (score) network.
    
    num_units_task: [1000, 1000]		# Hidden layer sizes of the task network.

  optimiser:
    learning_rate: 0.001		# Neural network learning rate.

    batch_size: 256		# Number of samples per training batch.

# ============================================================
# Output
# ============================================================

output:  
  directory: results/optimisation		# Directory for optimisation results.

  force: false	  # Whether to force an optimisation run to execute again.
```
### Overriding configuration settings

Any configuration setting can be overridden directly from the command line using:

```bash
edmri-optimise section.parameter=value
```

For example, to specify different input data, target data and superdesign:

```bash
edmri-optimise \
    data.input=my_signals.npy \
    data.target=my_targets.npy \
    data.superdesign=my_protocol.txt
```

Multiple settings can be overridden simultaneously. Command-line values take precedence over those in the configuration file.

### Example: changing the optimised protocol size

By default, TADRED retains 25% of the acquisitions in the superdesign.

To retain 10% instead:

```bash
edmri-optimise optimisation.protocol_size=0.1
```

To retain 50%:

```bash
edmri-optimise optimisation.protocol_size=0.5
```

### Example: changing training hyperparameters

To change the learning rate and batch size:

```bash
edmri-optimise \
    training.optimiser.learning_rate=0.0001 \
    training.optimiser.batch_size=512
```

To change the task network architecture to three hidden layers of 512 units:

```bash
edmri-optimise \
    training.network.num_units_task=[512,512,512]
```

## Application configuration

The `apply_default.yaml` file controls application of the trained task network to new MRI data.

It contains three main sections:

- **`data`:** New MRI measurements and an optional mask.
- **`task_network`:** Path to the trained task network produced during optimisation.
- **`output`:** Directory in which to save the predictions.

### Annotated `apply_default.yaml`

```yaml
# Default configuration for edmri-apply.

# ============================================================
# Input data
# ============================================================

data:
  # New MRI measurements acquired or simulated using
  # the optimised acquisition protocol.
  # Supports simulated data (.npy) or real MRI data (.nii.gz).
  input: examples/data/sandi/tadred_signals.npy

  # Optional mask for NIfTI data.
  mask: null

# ============================================================
# Trained task network
# ============================================================

# Network produced by edmri-optimise.
task_network: results/optimisation/examples/tadred_results/trained_task_network.pt

# ============================================================
# Output
# ============================================================

# Directory for the predicted task outputs.
output:
  directory: results/application
```

**Important:** The input data must contain the same acquisitions, in the same order, as the optimised protocol used to train the task network.

The optimised protocol itself does not need to be supplied to `edmri-apply`, because the task network has already been trained for that specific protocol.

### Example: applying a trained network to real MRI data

To apply the trained task network to a new NIfTI dataset:

```bash
edmri-apply \
    data.input=patient_data.nii.gz \
    data.mask=patient_mask.nii.gz \
    task_network=results/optimisation/<UNIQUE_HASH>/tadred_results/trained_task_network.pt
```

The trained network generates predictions for the specified task, such as quantitative MRI parameter maps.


## Using alternative configuration files

For experiments requiring substantially different settings, you can create a separate YAML configuration file.

For example, copy `optimise_default.yaml` to `my_experiment.yaml`, modify the settings and run:

```bash
edmri-optimise --config my_experiment.yaml
```

Similarly, an alternative application configuration can be specified using:

```bash
edmri-apply --config my_application.yaml
```

This allows experiments to maintain separate, reproducible configurations.

## Example mode

Both commands support the `--example` flag:

```bash
edmri-optimise --example
edmri-apply --example
```

This saves results to fixed example directories rather than configuration-specific hash directories, making the bundled examples easier to run.

For regular experiments, omit `--example` to keep results from different configurations separate.

See [detailed guide](detailed_guide.md) for a complete demonstration of the EDMRI workflow.