import traci

traci.start(["/Users/yihanhong/sumo/bin/sumo-gui", "-c", "sim.sumocfg"])

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    for vid in traci.vehicle.getIDList():
        pos = traci.vehicle.getPosition(vid)
        spd = traci.vehicle.getSpeed(vid)
        print(f"{vid}: pos={pos}, speed={spd:.1f} m/s")

traci.close()