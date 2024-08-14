from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True}) # we can also run as headless.
from sheeprl.cli import run

if __name__ == "__main__":
    run()
    simulation_app.close()
