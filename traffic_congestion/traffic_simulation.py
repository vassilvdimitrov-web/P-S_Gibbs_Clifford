import random
from node import ExitNode, EntryNode
from metrics import entry_density

safe_distance = 4
entry_rate = 0.05
dt = 0.01                                    # how much time passes between updates, time step
v_max = 30
max_cars_at_beginning = 10
density_window   = 30.0                      # metres from edge start used for entry-density check
max_edge_density = 0.1                       # cars per unit length within window; transfers blocked above this

class Car:
    def __init__(self, position, velocity, length, reaction_speed, max_accel, route):
        self.x = position
        self.v = velocity
        self.l = length
        self.reac = reaction_speed
        self.max_accel = max_accel
        self.route          = route   # list of Edge objects to traverse
        self.route_idx      = 0       # index of the edge the car is currently on
        self.waiting_to_exit = False  # parked at end of last edge until despawned

# change speed depending on the distance to the car in front
def update_velocities(cars):
    cars.sort(key=lambda car: car.x)
    for i, car in enumerate(cars):
        if car.waiting_to_exit:
            car.v = 0   # stationary obstacle — block cars behind
            continue
        if i + 1 < len(cars):
            next_car = cars[i + 1]
            gap = (next_car.x - car.x) - next_car.l
            if gap < safe_distance:
                car.v = max(0, car.v - 5 * car.reac)
                continue
        # last moving car on edge — no one ahead, accelerate freely
        car.v = min(v_max, car.v + car.max_accel * car.reac)

# x(t + dt) = x(t) + v*dt  — clamped so cars stop at the edge end
def update_positions(cars, road_length):
    for car in cars:
        if not car.waiting_to_exit:
            car.x = min(car.x + car.v * dt, road_length)

def spawn_car(entry_node_idx, node_types, routes):
    """
    Try to spawn a car at an entry node heading to a random reachable exit node.
    Returns (car, first_edge) if a valid route exists, otherwise None.
    Cars that complete their route (reach the exit node) are dropped by the caller.
    """
    reachable = routes.get(entry_node_idx, {})
    exit_dsts = [(dst, node_types[dst]) for dst in reachable if isinstance(node_types.get(dst), ExitNode)]
    if not exit_dsts:
        return None
    dsts    = [dst  for dst, _  in exit_dsts]
    weights = [node.demand for _, node in exit_dsts]
    dst     = random.choices(dsts, weights=weights, k=1)[0]
    route = reachable[dst]['path']
    if not route:
        return None
    return Car(0, random.uniform(5, v_max), 5, 0.2, 2.0, route), route[0]

def tick(node_types, routes, edges, nodes):
    """
    One simulation tick: spawn cars at entry nodes and advance all cars
    along their routes by 10 sub-steps.
    """

    # Spawn cars at entry nodes, routing only to reachable exit nodes
    for node_idx, node_type in node_types.items():
        if isinstance(node_type, EntryNode):
            if random.random() < node_type.spawn_probability:
                result = spawn_car(node_idx, node_types, routes)
                if result:
                    car, first_edge = result
                    first_edge.cars.append(car)

    # Simulate and advance cars along their routes
    for _ in range(10):
        graduated = []
        for edge in edges:
            exit_nt = node_types.get(edge.bezier.node1)
            despawn_prob = exit_nt.despawn_probability if isinstance(exit_nt, ExitNode) else None
            graduated += update_edge(edge.cars, edge.bezier.total_length(nodes), despawn_prob)
        
        for car in graduated:
            car.route_idx += 1
            if car.route_idx < len(car.route):
                next_edge = car.route[car.route_idx]
                if entry_density(next_edge.cars, density_window) >= max_edge_density:
                    # Destination too dense — hold car just below the graduation threshold
                    # so it stays in edge.cars, blocks cars behind it, and retries next sub-step
                    car.route_idx -= 1
                    current_edge = car.route[car.route_idx]
                    car.x = current_edge.bezier.total_length(nodes) - 1.5
                    car.v = 0
                    current_edge.cars.append(car)
                else:
                    car.x = 0
                    next_edge.cars.append(car)
            else:
                # Park at the end of the last edge; despawned probabilistically each tick
                last_edge = car.route[-1]
                car.waiting_to_exit = True
                car.v = 0
                car.x = last_edge.bezier.total_length(nodes) - 1.5
                last_edge.cars.append(car)


def update_edge(cars, road_length, exit_despawn_prob=None):
    """
    Advance cars on one edge by one time step.
    Waiting cars (parked at exit) are despawned probabilistically.
    Returns the list of cars that reached the end and need to move to the next edge.
    """
    if exit_despawn_prob is not None:
        cars[:] = [c for c in cars
                   if not (c.waiting_to_exit and random.random() < exit_despawn_prob)]

    graduated = [c for c in cars if not c.waiting_to_exit and c.x >= road_length - 1]
    cars[:]   = [c for c in cars if  c.waiting_to_exit or   c.x <  road_length - 1]

    if cars:
        update_velocities(cars)
        update_positions(cars, road_length)

    return graduated
