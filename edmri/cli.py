import argparse
import shutil
from pathlib import Path

from omegaconf import OmegaConf
import edmri
from tadred import utils

from .optimise import optimise_experiment
from .apply import apply_task_network
from .helpers import get_config_hash

def resolve_path(path, root):
    if path is None:
        return None

    path = Path(path).expanduser()
    return path if path.is_absolute() else root / path

def optimise_cli():
    """Command-line interface for experiment optimisation."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="edmri/conf/optimise_default.yaml",
        help="Path to YAML configuration file.",
    )

    # parser.add_argument(
    #     "--example",
    #     action="store_true",
    #     help="Run in example mode, using a fixed output directory instead of a configuration hash.",
    # )
    
    parser.add_argument(
        "--example",
        nargs="?",
        const="test",
        default=None,
        metavar="SUBFOLDER",
        help="Run in example mode, saving to examples/SUBFOLDER (default: examples/test).",
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run of experiment even if results already exist.",
    )

    arg, overrides = parser.parse_known_args()
    

    #locate the project root (parent of edmri package)
    project_root = Path(edmri.__file__).resolve().parent.parent

    project_root = Path(edmri.__file__).resolve().parent.parent

    config_path = Path(arg.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    config = OmegaConf.merge(
        OmegaConf.load(config_path),
        OmegaConf.from_dotlist(overrides),
    )

    # set the root path in the config to the project root
    if config.paths.root is None:
        config.paths.root = str(project_root)

    #make a hash of the config to use as a unique identifier for the experiment
    config_hash = None if arg.example is not None else get_config_hash(config)

    experiment_dir = project_root / config.output.directory / (
        Path("examples") / arg.example
        if arg.example is not None
        else config_hash
    )
    
    
    completed_flag_path = experiment_dir / "completed.flag"


    # Check whether the experiment has already completed
    if completed_flag_path.exists() and not arg.force:
        print(
            f"Experiment already completed: {experiment_dir}\n"
            "Use --force to re-run the experiment."
        )
        return

    # Remove existing results if forcing a rerun or recovering
    # from an incomplete experiment
    if experiment_dir.exists():
        if completed_flag_path.exists():
            print(f"Forcing re-run of experiment: {experiment_dir}")
        else:
            print(f"Removing incomplete experiment: {experiment_dir}")

        shutil.rmtree(experiment_dir)

    #make the output directory for the experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save the configuration to the experiment directory
    OmegaConf.save(config,experiment_dir / "config.yaml",resolve=True)

    #set up the tadred arguments from the config
    tadred_args = utils.load_base_args()

    tadred_args.network.num_units_score = config.training.network.num_units_score
    tadred_args.network.num_units_task = config.training.network.num_units_task
    
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
        tadred_args=tadred_args,
        force=arg.force,
    )

    # create a empty flag file to indicate that the experiment has completed successfully
    completed_flag_path.touch()


def apply_cli():
    """Command-line interface for applying a trained model."""

    parser = argparse.ArgumentParser(
        description="Apply a trained TADRED model to MRI data."
    )    
    
    parser.add_argument(
        "--config",
        default="edmri/conf/apply_default.yaml",
        help="Path to YAML configuration file.",
    )
    
    parser.add_argument(
        "--example",
        nargs="?",
        const="test",
        default=None,
        metavar="SUBFOLDER",
        help="Run in example mode, saving to examples/SUBFOLDER (default: examples/test).",
    )
        
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run of experiment even if results already exist.",
    )

    arg, overrides = parser.parse_known_args()

    #locate the project root (parent of edmri package)
    project_root = Path(edmri.__file__).resolve().parent.parent

    config_path = Path(arg.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    config = OmegaConf.merge(
        OmegaConf.load(config_path),
        OmegaConf.from_dotlist(overrides),
    )

    # set the root path in the config to the project root
    if config.paths.root is None:
        config.paths.root = str(project_root)


    #make a hash of the config to use as a unique identifier for the experiment
    config_hash = None if arg.example is not None else get_config_hash(config)

    experiment_dir = project_root / config.output.directory / (
            Path("examples") / arg.example
            if arg.example is not None
            else config_hash
        )
        
 
    
    completed_flag_path = experiment_dir / "completed.flag"

    if completed_flag_path.exists() and not arg.force:
        print(
            f"Application of task network experiment already completed: {experiment_dir}"
            f"\nUse --force to re-run the experiment."
        )
        return 
    

    # Remove existing results if forcing a rerun or recovering
    # from an incomplete experiment
    if experiment_dir.exists():
        if completed_flag_path.exists():
            print(f"Forcing re-run of experiment: {experiment_dir}")
        else:
            print(f"Removing incomplete experiment: {experiment_dir}")

        shutil.rmtree(experiment_dir)

    #make the output directory for the experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save the configuration to the experiment directory
    OmegaConf.save(config,experiment_dir / "config.yaml",resolve=True)

    #run experiment applying the task network to the input data
    apply_task_network(
        input_data=config.data.input,
        trained_task_network=config.task_network,
        mask=config.data.get("mask", None),
        output_file=experiment_dir / "predictions.npy",
    )

    # create a empty flag file to indicate that the experiment has completed successfully
    completed_flag_path.touch()