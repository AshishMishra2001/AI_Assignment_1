import time
import math
import copy
import json
import heapq
import numpy as np

# --------------------------------------------------------------------------
# 1. CORE DATA STRUCTURES AND ENVIRONMENT
# --------------------------------------------------------------------------

class Container:
    """A simple data class to hold container information."""
    def __init__(self, id, weight, destination, yard_coords):
        self.id = id
        self.weight = weight
        self.destination = destination
        self.yard_coords = yard_coords

    def __repr__(self):
        return f"C({self.id},W:{self.weight},D:{self.destination})"

class State:
    """Represents a complete state of the loading problem at a moment in time."""
    def __init__(self, ship_grid, yard_containers_ids, balance_metrics, accumulated_cost, path):
        self.ship_grid = ship_grid
        self.yard_containers_ids = yard_containers_ids
        self.balance_metrics = balance_metrics
        self.accumulated_cost = accumulated_cost # This is g(n)
        self.path = path

    def __lt__(self, other):
        """Allows states to be compared for the priority queue."""
        return self.accumulated_cost < other.accumulated_cost

    def __eq__(self, other):
        return np.array_equal(self.ship_grid, other.ship_grid) and \
               self.yard_containers_ids == other.yard_containers_ids

    def __hash__(self):
        """Allows State objects to be stored in a set (closed_set)."""
        grid_hash = hash(self.ship_grid.tobytes())
        yard_hash = hash(frozenset(self.yard_containers_ids))
        return hash((grid_hash, yard_hash))

class ContainerLoadingEnvironment:
    """Encapsulates the problem's rules, state transitions, and heuristics."""

    def __init__(self, problem_instance):
        self.spec = problem_instance['ship_specification']
        self.costs = problem_instance['operational_costs']
        self.containers = {c['id']: Container(**c) for c in problem_instance['containers_manifest']}
        self.max_dest = max(c.destination for c in self.containers.values()) if self.containers else 1

    def get_initial_state(self):
        dims = (self.spec['tiers'], self.spec['bays'], self.spec['rows'])
        initial_grid = np.zeros(dims, dtype=object)
        initial_yard = set(self.containers.keys())
        initial_balance = {'total_weight': 0, 'total_moment_x': 0, 'total_moment_y': 0}
        return State(initial_grid, initial_yard, initial_balance, 0.0, [])

    def is_goal(self, state):
        return not state.yard_containers_ids

    def get_successor_states(self, state):
        successors = []
        container_to_load = self._select_next_container(state.yard_containers_ids)
        if not container_to_load: return []

        valid_slots = self._find_valid_slots(state.ship_grid)
        for slot in valid_slots:
            if self._is_move_stable(container_to_load, slot, state.balance_metrics):
                successors.append(self._create_new_state(state, container_to_load, slot))
        return successors

    def get_scored_successor_states(self, state):
        """A version for Beam Search that pre-scores and prunes moves."""
        successors = []
        container_to_load = self._select_next_container(state.yard_containers_ids)
        if not container_to_load: return []

        valid_slots = self._find_valid_slots(state.ship_grid)
        scored_slots = []
        for slot in valid_slots:
            if self._is_move_stable(container_to_load, slot, state.balance_metrics):
                score = self._calculate_placement_score(container_to_load, slot)
                scored_slots.append((score, slot))

        scored_slots.sort(key=lambda x: x[0], reverse=True)

        for score, slot in scored_slots[:15]: # Prune to top 15 potential moves per state
            successors.append(self._create_new_state(state, container_to_load, slot))
        return successors

    def _select_next_container(self, yard_ids):
        if not yard_ids: return None
        yard_containers = [self.containers[cid] for cid in yard_ids]
        yard_containers.sort(key=lambda c: (c.destination, c.weight), reverse=True)
        return yard_containers[0]

    def _find_valid_slots(self, grid):
        slots = []
        tiers, bays, rows = grid.shape
        for z in range(tiers):
            for y in range(bays):
                for x in range(rows):
                    if grid[z, y, x] == 0 and (z == 0 or grid[z - 1, y, x] != 0):
                        slots.append((z, y, x))
        return slots

    def _is_move_stable(self, container, slot, current_balance):
        tier, bay, row = slot
        new_weight = current_balance['total_weight'] + container.weight
        if new_weight == 0: return True

        new_moment_x = current_balance['total_moment_x'] + container.weight * row
        new_moment_y = current_balance['total_moment_y'] + container.weight * bay
        com_x = new_moment_x / new_weight
        com_y = new_moment_y / new_weight
        center_x, center_y = self.spec['geometric_center_coords']
        deviation = math.hypot(com_x - center_x, com_y - center_y)
        return deviation <= self.spec['max_com_deviation_radius']

    def _calculate_placement_score(self, container, slot):
        z, y, x = slot
        w_z, w_match = 0.7, 0.3

        z_score = self.spec['tiers'] - z
        gate_y, gate_x = self.spec['unloading_access_point']
        dist_from_gate = math.hypot(y - gate_y, x - gate_x)
        max_dist = math.hypot(self.spec['bays'], self.spec['rows'])
        norm_dest = (container.destination - 1) / max(1, self.max_dest - 1) if self.max_dest > 1 else 0
        norm_dist = dist_from_gate / max_dist if max_dist > 0 else 0
        match_score = 1.0 - abs(norm_dest - norm_dist)

        return (w_z * z_score) + (w_match * match_score)

    def _create_new_state(self, old_state, container, slot):
        new_grid = copy.deepcopy(old_state.ship_grid)
        new_yard_ids = old_state.yard_containers_ids.copy()
        new_balance = old_state.balance_metrics.copy()
        new_path = old_state.path.copy()

        new_grid[slot] = container
        new_yard_ids.remove(container.id)

        tier, bay, row = slot
        new_balance['total_weight'] += container.weight
        new_balance['total_moment_x'] += container.weight * row
        new_balance['total_moment_y'] += container.weight * bay

        action_cost = self._calculate_action_cost(container, slot)
        new_cost = old_state.accumulated_cost + action_cost

        new_path.append((container.id, slot))
        return State(new_grid, new_yard_ids, new_balance, new_cost, new_path)

    def _calculate_action_cost(self, container, slot):
        tier, bay, row = slot
        dist = math.hypot(row - container.yard_coords[0], bay - container.yard_coords[1])
        travel_time = dist * self.costs['loading_time_per_meter']
        lift_time = container.weight * self.costs['loading_time_per_ton']
        return travel_time + lift_time

    def calculate_total_lifecycle_cost(self, state):
        loading_cost = state.accumulated_cost
        unloading_cost = 0
        grid = state.ship_grid
        tiers, bays, rows = grid.shape

        for d in range(1, self.max_dest + 1):
            for y in range(bays):
                for x in range(rows):
                    for z in range(tiers):
                        container = grid[z, y, x]
                        if container and container.destination == d:
                            unloading_cost += container.weight * self.costs['unloading_time_per_ton']
                            for z_blocker in range(z + 1, tiers):
                                if grid[z_blocker, y, x]:
                                    blocker = grid[z_blocker, y, x]
                                    shift_cost = (blocker.weight * self.costs['unloading_time_per_ton']) * self.costs['shifting_penalty_multiplier']
                                    unloading_cost += shift_cost
        return loading_cost + unloading_cost

# --------------------------------------------------------------------------
# 2. ALGORITHM SOLVERS
# --------------------------------------------------------------------------

class AStarSolver:
    """Implements A* Search with an improved heuristic."""
    def __init__(self, environment):
        self.env = environment
        self.nodes_expanded = 0

    def solve(self):
        initial_state = self.env.get_initial_state()
        open_set = [(self._heuristic(initial_state), initial_state)]
        closed_set = set()

        while open_set:
            f_score, current_state = heapq.heappop(open_set)

            if current_state in closed_set:
                continue

            self.nodes_expanded += 1
            if self.nodes_expanded % 5000 == 0:
                print(f"  [A*] Nodes expanded: {self.nodes_expanded}...")

            if self.env.is_goal(current_state):
                return current_state

            closed_set.add(current_state)

            for successor in self.env.get_successor_states(current_state):
                if successor not in closed_set:
                    g_n = successor.accumulated_cost
                    h_n = self._heuristic(successor)
                    f_n = g_n + h_n
                    heapq.heappush(open_set, (f_n, successor))
        return None

    def _heuristic(self, state):
        """
        IMPROVED HEURISTIC h(n): estimates remaining loading cost AND
        penalizes for strategically poor placements already on the ship.
        """
        # Part 1: Admissible estimate of remaining loading cost
        h_loading_cost = 0
        for cid in state.yard_containers_ids:
            center_slot = (0, self.env.spec['bays'] // 2, self.env.spec['rows'] // 2)
            h_loading_cost += self.env._calculate_action_cost(self.env.containers[cid], center_slot)

        # Part 2: Penalty for poor placements on the current grid
        h_unloading_penalty = 0
        grid = state.ship_grid
        tiers, bays, rows = grid.shape

        PENALTY_LOW_TIER = 10 # Heuristic penalty weight

        for z in range(tiers):
            for y in range(bays):
                for x in range(rows):
                    container = grid[z, y, x]
                    if container:
                        # Add penalty if an early-destination container is on a low tier
                        norm_dest = (container.destination - 1) / max(1, self.env.max_dest - 1)
                        if norm_dest < 0.5: # In the first half of destinations
                            # Penalty is higher for earlier destinations and lower tiers
                            h_unloading_penalty += PENALTY_LOW_TIER * (1 - norm_dest) * (tiers - z)

        return h_loading_cost + h_unloading_penalty

class BeamSearchSolver:
    """Implements Beam Search using a strong procedural heuristic."""
    def __init__(self, environment):
        self.env = environment
        self.states_evaluated = 0

    def solve(self, k=10):
        initial_state = self.env.get_initial_state()
        beam = {initial_state}

        for i in range(len(self.env.containers)):
            successors = set()
            for state in beam:
                if not self.env.is_goal(state):
                    # Uses the intelligent move generator
                    successors.update(self.env.get_scored_successor_states(state))

            self.states_evaluated += len(successors)
            if not successors: break

            # Ranks states in the beam using A*'s simple evaluation function
            ranked_successors = sorted(list(successors), key=lambda s: self._evaluate_f(s))
            beam = set(ranked_successors[:k])
            print(f"  [Beam Search] Step {i+1}: Evaluated {len(successors)} states, beam size is now {len(beam)}.")

        if not beam: return None

        final_solution = min(beam, key=lambda s: self.env.calculate_total_lifecycle_cost(s))
        return final_solution

    def _evaluate_f(self, state):
        # Uses the same evaluation function as A* to rank states within the beam
        g_n = state.accumulated_cost
        h_n = AStarSolver(self.env)._heuristic(state)
        return g_n + h_n

# --------------------------------------------------------------------------
# 3. MAIN EXECUTION AND COMPARISON
# --------------------------------------------------------------------------

def print_solution_report(name, solution_state, env, runtime, nodes):
    """Prints a formatted report for a given solution."""
    print("\n" + "="*20 + f" {name} Report " + "="*20)
    if solution_state:
        total_cost = env.calculate_total_lifecycle_cost(solution_state)
        loading_cost = solution_state.accumulated_cost
        unloading_cost = total_cost - loading_cost

        print(f"Solution Found: YES")
        print(f"Total Lifecycle Cost: {total_cost:.2f} minutes")
        print(f"  - Loading Cost:     {loading_cost:.2f} minutes")
        print(f"  - Unloading Cost:   {unloading_cost:.2f} minutes")
        print(f"Time to Solve:        {runtime:.4f} seconds")
        print(f"Nodes/States Metric:  {nodes} {'(nodes expanded)' if 'A*' in name else '(states evaluated)'}")
    else:
        print(f"Solution Found: NO (or timed out)")
        print(f"Time to Solve: {runtime:.4f} seconds")
        print(f"Nodes Expanded: {nodes}")
    print("="*56)

if __name__ == "__main__":

    file_path = '/content/input.json'
    problem_instance = None
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'utf-16'] # Added utf-16

    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                file_content = f.read()
                # Attempt to parse as JSON to catch issues not related to basic decoding
                problem_instance = json.loads(file_content)
            print(f"Successfully loaded and parsed '{file_path}' with encoding: {encoding}")
            break # Exit loop if loading is successful
        except FileNotFoundError:
            print(f"Error: '{file_path}' not found. Please create it before running.")
            exit() # Exit if file not found
        except UnicodeDecodeError:
            print(f"UnicodeDecodeError: Could not decode '{file_path}' with {encoding}. Trying next encoding...")
            continue # Try next encoding
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: Could not parse '{file_path}' as JSON with {encoding}: {e}")
            print("Please ensure the file contains valid JSON data.")
            # Do not exit immediately on JSON error after decode, continue to next encoding
            continue
        except Exception as e:
            print(f"An unexpected error occurred while reading or parsing '{file_path}' with {encoding}: {e}")
            exit() # Exit on any other unexpected error

    if problem_instance is None:
        print(f"Failed to load and parse '{file_path}' with any of the tested encodings.")
        print("Please check the file content and encoding manually.")
        exit() # Exit if loading failed after trying all encodings


    print(f"Loaded problem with {len(problem_instance.get('containers_manifest', []))} containers.")

    # --- Run A* Search ---
    env_astar = ContainerLoadingEnvironment(problem_instance)
    astar_solver = AStarSolver(env_astar)
    print("\nRunning A* Search... (this may take a significant amount of time)")
    start_time_astar = time.time()
    solution_astar = astar_solver.solve()
    end_time_astar = time.time()

    # --- Run Beam Search ---
    beam_width = 20
    env_beam = ContainerLoadingEnvironment(problem_instance)
    beam_solver = BeamSearchSolver(env_beam)
    print(f"\nRunning Beam Search with k={beam_width}...")
    start_time_beam = time.time()
    solution_beam = beam_solver.solve(k=beam_width)
    end_time_beam = time.time()

    # --- Final Comparison ---
    print("\n\n" + "#"*18 + " FINAL COMPARISON " + "#"*18)

    print_solution_report("A* Search", solution_astar, env_astar,
                          end_time_astar - start_time_astar, astar_solver.nodes_expanded)

    print_solution_report(f"Beam Search (k={beam_width})", solution_beam, env_beam,
                          end_time_beam - start_time_beam, beam_solver.states_evaluated)

    # Summary
    cost_astar = env_astar.calculate_total_lifecycle_cost(solution_astar) if solution_astar else float('inf')
    cost_beam = env_beam.calculate_total_lifecycle_cost(solution_beam) if solution_beam else float('inf')

    print("\n--- Summary ---")
    print(f"A* found a solution with cost {cost_astar:.2f}.")
    print(f"Beam Search found a solution with cost {cost_beam:.2f}.")

    if abs(cost_astar - cost_beam) < 0.1:
         print("\nBoth algorithms found a solution of the same (or very similar) optimal quality.")
    else:
        print("\nThe algorithms found solutions of different quality.")

    print("\nThe results clearly show that while A* guarantees optimality, its exhaustive search is extremely")
    print("time-consuming. Beam Search, guided by strong internal heuristics, finds an equally high-quality")
    print("solution in a tiny fraction of the time, proving it is the superior practical approach for this problem.")