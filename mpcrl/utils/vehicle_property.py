from highway_env.vehicle.kinematics import Vehicle

class VehicleSetting:
    """
    A collection of all the vehicle settings,
    useful for rendering and visualization.
    """
    
    # vehicle dimension
    width = Vehicle.WIDTH
    length = Vehicle.LENGTH
    
    # color: 
    ego_color = "green"
    agent_color = "blue"
    conflict_agent_color = "red"
    trajectory_color = "grey"