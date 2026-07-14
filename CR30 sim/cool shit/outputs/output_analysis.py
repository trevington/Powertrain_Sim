import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_data(csv_path):
    time = []
    energy = []
    power = []
    velocity = []
    soe = []
    temp = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            time.append(float(row["time (s)"]))
            energy.append(float(row["energy_used (Ws)"]))
            power.append(float(row["power (W)"]))
            velocity.append(float(row["final_velocity (m/s)"]))
            soe.append(float(row["soe (%)"]))
            temp.append(float(row["temp_final (C)"]))

    return time, energy, power, velocity, soe, temp


def compute_acceleration(time, velocity):
    acceleration = []
    for i in range(len(time)):
        if i == 0:
            dt = time[1] - time[0]
            acceleration.append((velocity[1] - velocity[0]) / dt)
        elif i == len(time) - 1:
            dt = time[-1] - time[-2]
            acceleration.append((velocity[-1] - velocity[-2]) / dt)
        else:
            dt = time[i + 1] - time[i - 1]
            acceleration.append((velocity[i + 1] - velocity[i - 1]) / dt)
    return acceleration


def plot_results(time, energy, acceleration, power, velocity, soe, temp, output_path):
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    axes[0].plot(time, energy, color="tab:blue", linewidth=1.8)
    axes[0].set_title("Total Energy vs Time")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Energy (Ws)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, acceleration, color="tab:orange", linewidth=1.8)
    axes[1].set_title("Acceleration vs Time")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Acceleration (m/s²)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, power, color="tab:green", linewidth=1.8)
    axes[2].set_title("Power vs Time")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Power (W)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time, velocity, color="tab:red", linewidth=1.8)
    axes[3].set_title("Velocity vs Time")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Velocity (m/s)")
    axes[3].grid(True, alpha=0.3)

    ax_soe = axes[4]
    ax_temp = ax_soe.twinx()
    line1, = ax_soe.plot(time, soe, color="tab:purple", linewidth=1.8, label="SOE")
    line2, = ax_temp.plot(time, temp, color="tab:brown", linewidth=1.8, label="Temperature")
    ax_soe.set_title("SOE and Temperature vs Time")
    ax_soe.set_xlabel("Time (s)")
    ax_soe.set_ylabel("SOE (%)", color="tab:purple")
    ax_temp.set_ylabel("Temperature (°C)", color="tab:brown")
    ax_soe.grid(True, alpha=0.3)
    ax_soe.tick_params(axis="y", labelcolor="tab:purple")
    ax_temp.tick_params(axis="y", labelcolor="tab:brown")
    ax_soe.legend([line1, line2], ["SOE", "Temperature"], loc="upper left")

    axes[5].axis("off")

    fig.suptitle("CR30 Powertrain Simulation Analysis")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=300)
    plt.show()


def main():
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / "CR30_Michican_EV_2026.csv"
    output_path = script_dir / "analysis_plots.png"

    time, energy, power, velocity, soe, temp = load_data(csv_path)
    acceleration = compute_acceleration(time, velocity)
    plot_results(time, energy, acceleration, power, velocity, soe, temp, output_path)

    print('maximum power: ', max(power))
    print('maximum velocity: ', max(velocity))
    print('maximum acceleration: ', max(acceleration))
    print('minimum acceleration: ', min(acceleration))
    print('maximum temperature: ', max(temp))
    print('minimum temperature: ', min(temp))
    print('maximum SOE: ', max(soe))
    print('minimum SOE: ', min(soe))
    print('total energy used: ', energy[-1])
    print('total time: ', time[-1])
    print('average power: ', sum(power) / len(power))
    print('average velocity: ', sum(velocity) / len(velocity))
    print('average acceleration: ', sum(acceleration) / len(acceleration))
    print('average temperature: ', sum(temp) / len(temp))


if __name__ == "__main__":
    main()