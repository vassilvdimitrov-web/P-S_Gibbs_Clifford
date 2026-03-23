import random
import numpy as np

class EntryNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.waiting_queue = []
        # Traffic light states
        self.entry_green = False
        self.main_green = True
        self.timer = 0
        self.phase_duration = random.randint(150, 300)

class Car:
    def __init__(self, current_edge, velocity, reaction_speed, destination_node_idx, length=4.5, is_bad_driver=False):
        self.current_edge = current_edge 
        self.x = 0.0 
        self.v = velocity
        self.reac = reaction_speed
        self.l = length
        self.destination_node_idx = destination_node_idx
        self.is_bad_driver = is_bad_driver
        self.is_waiting_to_enter = True

def can_safely_enter(target_edge, active_cars, safe_distance=0.5):
    """
    Checks if the start of a specific edge is clear.

    """
    for car in active_cars:
        # We only care about cars ALREADY on the edge we want to join
        if car.current_edge == target_edge:
            # If a car is within the safe distance from the start of the curve
            if car.x < safe_distance:
                return False
    return True

def process_node_entries(entry_nodes, active_cars):
    """
    Moves cars from queue to road ONLY if the entry light is green
    AND there is enough space to merge safely.
    """
    for node in entry_nodes:
        # Step 1: Is someone waiting AND is the light green?
        if len(node.waiting_queue) > 0 and node.entry_green: 
            next_car = node.waiting_queue[0]
            
            # Step 2: Is there a gap in the ring road traffic?
            if can_safely_enter(next_car.current_edge, active_cars, safe_distance=25.0):
                entering_car = node.waiting_queue.pop(0)
                entering_car.is_waiting_to_enter = False
                entering_car.x = 0.0
                active_cars.append(entering_car)

                
def handle_edge_transitions(cars, edges, nodes):
    remaining_cars = []
    for car in cars:
        # 1. Use the bezier attribute to get length
        edge_len = car.current_edge.bezier.total_length(nodes)
        
        if car.x >= edge_len:
            # 2. Check destination using .bezier.node1
            if car.current_edge.bezier.node1 == car.destination_node_idx:
                continue # Car successfully exited the system
            
            # 3. Find next options looking at .bezier.node0 and .bezier.node1
            next_options = [e for e in edges if e.bezier.node0 == car.current_edge.bezier.node1]
            
            if next_options:
                car.current_edge = random.choice(next_options)
                car.x = 0
                remaining_cars.append(car)
            else:
                # Dead end - car removes itself
                pass 
        else:
            remaining_cars.append(car)
    return remaining_cars

def generate_entry_demand(entry_nodes, edges, node_types, probability=0.05):
    """
    Adds cars to node queues and assigns them a starting edge and destination.
    """
    for node in entry_nodes:
        if random.random() < probability:
            # Find edges starting at this node
            outgoing = [e for e in edges if e.bezier.node0 == node.node_id]
            if outgoing:
                start_edge = random.choice(outgoing)
                # Pick a random destination node index
                exit_nodes = []
                weights = []

                for i, t in node_types.items():
                    if t.__class__.__name__ == "ExitNode":
                        exit_nodes.append(i)
                        weights.append(max(0.0001, t.demand))  # avoid zero weight

                if not exit_nodes:
                    continue

                dest = random.choices(exit_nodes, weights=weights, k=1)[0]
                #new_car = Car(start_edge, random.uniform(2, 5), random.uniform(0.5, 1.5), dest)
                is_bad = random.random() < 0.10   # 15% bad drivers
                new_car = Car(
                    start_edge,
                    random.uniform(2, 5),
                    random.uniform(0.5, 1.5),
                    dest,
                    is_bad_driver=is_bad
                )
                node.waiting_queue.append(new_car)

def update_traffic_lights(entry_nodes):
    for node in entry_nodes:
        node.timer += 1
        if node.timer > node.phase_duration:
            node.entry_green = not node.entry_green
            node.main_green = not node.main_green
            node.timer = 0

def apply_traffic_lights(cars, entry_nodes, nodes):
    for car in cars:
        target_node_id = car.current_edge.bezier.node1
        for enode in entry_nodes:
            if enode.node_id == target_node_id:
                edge_len = car.current_edge.bezier.total_length(nodes)
                dist_to_light = edge_len - car.x
                
                # Only stop if the car is close to the intersection (e.g., within 40 pixels)
                if dist_to_light < 40: 
                    if not enode.main_green:
                        car.v = 0  # Stop at the red light