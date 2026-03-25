import random

v_max = 50.0
safe_distance = 10
dt = 0.01                                    # how much time passes between updates, time step

bad_driver_safe_distance_factor = 1.5
bad_driver_acceleartion_factor = 0.7
bad_driver_safe_brake_factor = 1.5

default_acceleration = 3.0
default_brake_factor = 5.0

def local_speed_limit(x,edge=None):
    if edge is not None and hasattr(edge, "speed_limit"):
        return float(edge.speed_limit)
    return v_max

#return all the cars driving on this edge
def cars_on_edge(cars,edge):
    result = []
    for car in cars:
        if car.current_edge is edge and not car.is_waiting_to_enter:
            result.append(car)
    return result

def next_car_ahead(car, edge_cars):
    ahead = []
    for other in edge_cars:
        if other is not car and other.x > car.x:
            ahead.append(other)

    if len(ahead) == 0:
        return None

    leader = ahead[0]
    for other in ahead[1:]:
        if other.x < leader.x:
            leader = other
    return leader

def gap_to_next_car(car, leader):
    gap = leader.x - car.x - leader.l
    if (gap < 0.0):
        return 0.0
    return gap

def edge_length(edge, nodes):
    return float(edge.bezier.total_length(nodes))

#update velocities for all active cars on one edge.
#bad drivers leave larger gaps, accelerate less, and brake harde
def update_velocities_on_edge(edge, cars, nodes):
    edge_cars = cars_on_edge(cars, edge)
    edge_cars.sort(key=lambda c: c.x)

    for car in edge_cars:
        if getattr(car, "is_bad_driver", False):
            safe_dist = safe_distance * bad_driver_safe_distance_factor
            accel = default_acceleration * bad_driver_acceleartion_factor
            brake = default_brake_factor * bad_driver_safe_brake_factor
        else:
            safe_dist = safe_distance
            accel = default_acceleration
            brake = default_brake_factor

        leader = next_car_ahead(car, edge_cars)
        limit = local_speed_limit(car.x, edge)

        if leader is None:
            car.v = min(limit, car.v + accel)
        else:
            gap = gap_to_next_car(car, leader)

            if gap < safe_dist:
                car.v = max(0.0, car.v - brake * car.reac)
            else:
                car.v = min(limit, car.v + accel)

def update_positions_on_edge(edge, cars, nodes):       
    length = edge_length(edge, nodes)
    for car in cars:
        if car.current_edge is not edge:
            continue
        if car.is_waiting_to_enter:
            continue

        car.x += car.v * dt

        if car.x > length:
            car.x = length


#so we could use update_velocities(cars, nodes, edges) directly
def update_velocities(cars, nodes, edges=None):
    if edges is None:
        edges = []
        for car in cars:
            if not car.is_waiting_to_enter and car.current_edge is not None:
                if car.current_edge not in edges:
                    edges.append(car.current_edge)

    for edge in edges:
        update_velocities_on_edge(
            edge=edge,
            cars=cars,
            nodes=nodes,
        )
def update_positions(cars, nodes, edges=None):
    if edges is None:
        edges = []
        for car in cars:
            if not car.is_waiting_to_enter and car.current_edge is not None:
                if car.current_edge not in edges:
                    edges.append(car.current_edge)

    for edge in edges:
        update_positions_on_edge(
            edge=edge,
            cars=cars,
            nodes=nodes,
        )


