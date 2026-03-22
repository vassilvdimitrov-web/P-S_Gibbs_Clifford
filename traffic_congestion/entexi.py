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
    def __init__(self, current_edge, velocity, reaction_speed, destination_node_idx, length=4.5):
        self.current_edge = current_edge 
        self.x = 0.0 
        self.v = velocity
        self.reac = reaction_speed
        self.l = length
        self.destination_node_idx = destination_node_idx
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
    Checks all entry nodes. If a car is waiting and its path is clear,
    it moves from the queue to the active car list.
    """
    for node in entry_nodes:
        if node.waiting_queue and node.entry_green: 
            next_car = node.waiting_queue[0]
            if can_safely_enter(next_car.current_edge, active_cars):
                entering_car = node.waiting_queue.pop(0)
                entering_car.is_waiting_to_enter = False
                entering_car.x = 0.0  # <--- FORCE START AT BEGINNING
                active_cars.append(entering_car)

                
def handle_edge_transitions(cars, edges, nodes):
    """
    Logic for cars reaching the end of a Bezier curve.
    Either exits or picks a new edge.
    """
    remaining_cars = []
    for car in cars:
        edge_len = car.current_edge.total_length(nodes)
        
        if car.x >= edge_len:
            # Check if this node is the destination
            if car.current_edge.node1 == car.destination_node_idx:
                continue # Exit car (don't add to remaining)
            
            # Find next edges
            next_options = [e for e in edges if e.node0 == car.current_edge.node1]
            
            if next_options:
                car.current_edge = random.choice(next_options)
                car.x = 0
                remaining_cars.append(car)
            else:
                pass # Dead end, car exits
        else:
            remaining_cars.append(car)
    return remaining_cars

def generate_entry_demand(entry_nodes, edges, node_count, probability=0.05):
    """
    Adds cars to node queues and assigns them a starting edge and destination.
    """
    for node in entry_nodes:
        if random.random() < probability:
            # Find edges starting at this node
            outgoing = [e for e in edges if e.node0 == node.node_id]
            if outgoing:
                start_edge = random.choice(outgoing)
                # Pick a random destination node index
                dest = random.choice([i for i in range(node_count) if i != node.node_id])
                
                new_car = Car(start_edge, random.uniform(2, 5), random.uniform(0.5, 1.5), dest)
                node.waiting_queue.append(new_car)

def update_traffic_lights(entry_nodes):
    for node in entry_nodes:
        node.timer += 1
        if node.timer > node.phase_duration:
            node.entry_green = not node.entry_green
            node.main_green = not node.main_green
            node.timer = 0

def apply_traffic_lights(cars, entry_nodes, nodes):
    """
    Slows cars down if they are approaching a node with a red light.
    """
    for car in cars:
        # The node the car is approaching is node1 of its current edge
        target_node_id = car.current_edge.node1
        
        # Find the matching entry_node logic
        for enode in entry_nodes:
            if enode.node_id == target_node_id:
                edge_len = car.current_edge.total_length(nodes)
                dist_to_node = edge_len - car.x
                
                if 0 < dist_to_node < 60: # Visibility distance
                    if not enode.main_green:
                        car.v = max(0, car.v - 2 * car.reac)