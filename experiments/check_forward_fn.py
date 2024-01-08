"""Check forward fn of SE3 diffuser."""
import hydra
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import omegaconf

from experiments.inference import Inference

# pylint:disable=protected-access


def verify_rotation_forward_fn(
    inference: Inference,
    num_samples: int,
    t_forward_steps: np.ndarray,
    dt: float,
) -> None:
    """Verify the rotation forward function p(x(t) | x(t-1)).

    The forward function is verified by running forward "t" times
    and comparing the output to forward marginal (which samples
    p(x_t | x_0)). We run this comparison num_samples times and
    compare the distribution of angles produced by the forward
    and forward_marginal calls.

    Args:
        inference: FrameDiPT inference object.
        num_samples: number of times to sample x_t.
        t_forward_steps: array of timesteps.
        dt: time gap between two steps.
    """
    # Generate random unit vectors for the axis of rotation
    rot_0 = np.zeros((num_samples, 3))

    # Get final timestep
    t_final = t_forward_steps[-1]

    # Run forward marginal for the r3 diffuser.
    (
        rot_t_0_forward_marginal,
        _,
    ) = inference.diffuser._so3_diffuser.forward_marginal(
        rot_0=rot_0,
        t=t_final,
    )

    # Run for the equivalent number of times using a single forward step.
    rot_t_forward = rot_0
    for t in t_forward_steps[:-1]:
        rot_t_forward = inference.diffuser._so3_diffuser.forward(
            x_t_1=rot_t_forward,
            t_1=t,
            dt=dt,
        )

    # Calculate the angle of rotation
    rot_t_0_forward_marginal_angle = np.linalg.norm(rot_t_0_forward_marginal, axis=-1)
    rot_t_forward_angle = np.linalg.norm(rot_t_forward, axis=-1)

    # Plot the distributions
    plt.figure(figsize=(10, 6))

    # Plot data1
    plt.hist(
        rot_t_0_forward_marginal_angle,
        bins=50,
        alpha=0.5,
        label="Forward Marginal",
        color="blue",
    )

    # Plot data2
    plt.hist(
        rot_t_forward_angle, bins=50, alpha=0.5, label="Forward (t times)", color="red"
    )

    # Add labels and title
    plt.xlabel("Angle")
    plt.ylabel("Frequency")
    plt.title(
        "Distribution of Mean Coordinate Values for forward "
        "marginal and forward (Rotation only)"
    )

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


def verify_translation_forward_fn(
    inference: Inference,
    num_samples: int,
    t_forward_steps: np.ndarray,
    dt: float,
) -> None:
    """Verify the translation forward function p(x(t) | x(t-1)).

    The forward function is verified by running forward "t" times
    and comparing the output to forward marginal (which samples
    p(x_t | x_0)). We run this comparison num_samples times and
    compare the distribution of values produced by the forward
    and forward_marginal calls.

    Args:
        inference: FrameDiPT inference object.
        num_samples: number of times to sample x_t.
        t_forward_steps: array of timesteps.
        dt: time gap between two steps.
    """
    # x_0 is N samples of a single residue at (1, 1, 1) with
    # shape (N_samples, N_residue, 3).
    x_0 = np.ones((num_samples, 1, 3))

    # Get final timestep
    t_final = t_forward_steps[-1]

    # Run forward marginal for the r3 diffuser.
    (
        x_t_0_forward_marginal,
        _,
    ) = inference.diffuser._r3_diffuser.forward_marginal(
        x_0=x_0,
        t=t_final,
    )

    # Run for the equivalent number of times using a single forward step.
    x_t_forward = x_0
    for t in t_forward_steps[:-1]:
        x_t_forward = inference.diffuser._r3_diffuser.forward(
            x_t_1=x_t_forward,
            t_1=t,
            dt=dt,
            center=False,
        )

    # Plot the distributions
    plt.figure(figsize=(10, 6))

    # Plot the mean of the forward marginal coordinates.
    x_t_0_forward_marginal_mean = np.mean(x_t_0_forward_marginal, axis=-1)
    plt.hist(
        x_t_0_forward_marginal_mean,
        bins=100,
        alpha=0.5,
        label="Forward Marginal",
        color="blue",
    )

    # Plot the mean of the forward coordinates.
    x_t_forward_mean = np.mean(x_t_forward, axis=-1)
    plt.hist(
        x_t_forward_mean, bins=100, alpha=0.5, label="Forward (t times)", color="red"
    )

    plt.xlabel("Mean Coordinate Value")
    plt.ylabel("Frequency")
    plt.title(
        "Distribution of Mean Coordinate Values "
        "for forward marginal and forward (Translation only)"
    )

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


@hydra.main(version_base="1.3.1", config_path="../config", config_name="inference")
def verify_forward_process(cfg: omegaconf.DictConfig) -> None:
    """Verify the forward process of the SE3 diffuser."""
    np.random.seed(0)
    inference = Inference(cfg=cfg)

    # Number of samples (batch size)
    num_samples = 50000

    # Generate the timesteps
    num_t = inference._cfg.inference.diffusion.num_t
    min_t = inference._cfg.inference.diffusion.min_t
    t_forward_steps = np.linspace(min_t, 1.0, num_t)[: num_t // 2]

    # Calculate time gap between two steps
    dt = 1 / num_t

    # Verify the translation forward function
    verify_translation_forward_fn(
        inference=inference,
        num_samples=num_samples,
        t_forward_steps=t_forward_steps,
        dt=dt,
    )

    # Verify the rotation forward function
    verify_rotation_forward_fn(
        inference=inference,
        num_samples=num_samples,
        t_forward_steps=t_forward_steps,
        dt=dt,
    )


if __name__ == "__main__":
    verify_forward_process()  # pylint: disable=no-value-for-parameter
