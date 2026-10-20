import argparse
from pathlib import Path
from networkx import config
from omegaconf import OmegaConf

from .optimise import optimise_experiment

from .helpers import get_config_hash
from .helpers import load_existing_results

def optimise_cli():
    """Command-line interface for experiment optimisation."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run of experiment even if results already exist.",
    )

    arg, overrides = parser.parse_known_args()

    config = OmegaConf.merge(OmegaConf.load(arg.config), OmegaConf.from_dotlist(overrides))

    #make a hash of the config to use as a unique identifier for the experiment
    config_hash = get_config_hash(config)
    experiment_dir = Path(config.output.directory) / config_hash
    completed_flag_path = experiment_dir / "completed.flag"


    if completed_flag_path.exists() and not arg.force:
        print(
            f"Experiment already completed: {experiment_dir}"
            f"\nUse --force to re-run the experiment."
        )
        return 

        
    #make the output directory for the experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Run optimisation
    optimise_experiment(
        input_data=config.data.input,
        target_data=config.data.target,
        superdesign=config.data.superdesign,
        mask=config.data.get("mask", None),
        opt_protocol_size=config.optimisation.protocol_size,
        n_iterations_tadred=config.optimisation.n_iterations,
        random_state=config.optimisation.random_state,
        output_dir=experiment_dir,
        config_hash=config_hash,
        force=config.output.force,
    )

    # create a empty flag file to indicate that the experiment has completed successfully
    completed_flag_path.touch()

    # Save the configuration to the experiment directory
    OmegaConf.save(config,experiment_dir / "config.yaml",resolve=True)


def apply_cli():
    """Command-line interface for applying a trained model."""

    parser = argparse.ArgumentParser(
        description="Apply a trained TADRED model to MRI data."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input MRI data (.npy or NIfTI).",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model checkpoint (.pt).",
    )

    parser.add_argument(
        "--mask",
        default=None,
        help="Optional NIfTI mask.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output file (.npy or .nii.gz).",
    )

    args = parser.parse_args()

    apply_trained_model(
        input_data=args.input,
        trained_model=args.model,
        mask=args.mask,
        output_file=args.output,
    )