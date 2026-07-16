import numpy as np
import matplotlib.pyplot as plt


# Week05 과제 1 조건
CENTER = np.array([100.0, -100.0, -150.0])
RADIUS = 40.0
HEIGHT_OFFSET = 30.0
PERIOD = 3.0
DT = 0.01


def circular_trajectory(t):
    """시간 t에 따른 발끝 원형 궤적을 계산한다."""
    angle = 2.0 * np.pi * t / PERIOD

    x = CENTER[0] + RADIUS * np.cos(angle)
    y = CENTER[1] + RADIUS * np.sin(angle)
    z = CENTER[2] + HEIGHT_OFFSET * np.sin(2.0 * angle)

    return np.array([x, y, z])


def main():
    times = np.arange(0.0, PERIOD + DT, DT)
    trajectory = np.array(
        [circular_trajectory(t) for t in times]
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        linewidth=2,
        label="Foot trajectory",
    )

    ax.scatter(
        CENTER[0],
        CENTER[1],
        CENTER[2],
        marker="x",
        s=100,
        label="Center",
    )

    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        trajectory[0, 2],
        s=70,
        label="Start",
    )

    ax.set_title("Week05 - Circular Foot Trajectory")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    ax.set_xlim(50, 150)
    ax.set_ylim(-150, -50)
    ax.set_zlim(-190, -110)

    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()