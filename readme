# Container Loading Optimization – A Comparative Heuristic Search Solution

**AI Assignment 1 | Ashish Mishra | MT24020**

---

### 1. Overview

This assignment contains the solution to the complex challenge of optimizing container loading on a ship. Our goal is to determine a near-optimal 3D placement for each container to minimize the **Total Lifecycle Cost**, a metric that includes both the initial loading time and the subsequent unloading time at different ports.

My solution implements and compares two powerful search algorithms: **A\* Search** (for optimality) and **Beam Search** (for practical scenarios and high performance). The program reads a detailed problem instance from a JSON file and respects a sophisticated set of constraints, including dynamic 3D ship balance (using Center of Mass) and a multi-layered destination ordering rule. The final output is a complete loading plan with a detailed breakdown of costs and performance.

### 2. Problem Formulation

To tackle this computationally, we framed the problem using the classic state-space search model from Artificial Intelligence. The ship is represented as a full 3D grid, and the goal is to find the most efficient sequence of actions to load all containers.

- The ship is a **3D grid** of Bays (length), Rows (width), and Tiers (height).
- Each container has a `weight`, `destination port` (integer), and an initial `2D coordinate` in the yard.
- The task is to load all containers into the grid while respecting all constraints and minimizing a complex cost function.

#### 2.1 AI Problem Formulation (5-tuple)

The problem can be described as a standard search problem `(S, s₀, A, T, G)`:

*   **States (S):** Each state is a complete configuration of the loading operation.
    `s = (ship_grid, yard_containers, ship_balance_metrics, accumulated_cost, path)`
    *   `ship_grid`: A 3D NumPy array representing the placement of all containers.
    *   `yard_containers`: A set of container IDs remaining in the yard.
    *   `ship_balance_metrics`: Tracks the ship's Center of Mass (CoM).
    *   `accumulated_cost`: The total loading cost (`g(n)`) incurred so far.
    *   `path`: The sequence of actions taken to reach the current state.

*   **Initial State (s₀):**
    *   `ship_grid` is empty (filled with zeros).
    *   `yard_containers` contains all containers to be loaded.
    *   `accumulated_cost` is 0.

*   **Actions (A):** Selecting an unloaded container and placing it in a valid 3D slot.
    `a = Load(container, target_position(tier, bay, row))`

*   **Transition Model (T):** Applying an action updates the state by:
    *   Placing the container in the `ship_grid`.
    *   Removing the container from `yard_containers`.
    *   Updating the ship's balance metrics.
    *   Increasing the `accumulated_cost`.

*   **Goal Test (G):**
    *   All containers are loaded (`yard_containers` is empty).
    *   The path taken must have satisfied the hard **Balance Constraint** at every step.

#### 2.2 Constraints

The solution enforces both hard constraints (must be satisfied) and soft constraints (guided by the heuristic).

1.  **Balance Constraint (Hard):** The ship's Center of Mass (CoM) must remain within a predefined safety radius of the ship's geometric center at all times. Any move that violates this is illegal.
2.  **Destination Stacking Constraint (Hard):** For any two containers in the same vertical stack where `C_upper` is on top of `C_lower`, it must be that `C_upper.destination <= C_lower.destination`. This prevents direct physical blocking.
3.  **Placement Efficiency Constraint (Soft):** Containers for earlier ports should, "as far as possible," be placed closer to the ship's access point and in higher tiers. This is a strategic goal enforced by the search heuristic.

### 3. Cost Modeling

Our objective function minimizes the **Total Lifecycle Cost**.

`TotalLifecycleCost = LoadingCost + UnloadingCost`

*   **Loading Cost (`g(n)`):** The known path cost, calculated from the crane's travel time (distance-based) and lifting effort (weight-based).
    `ActionCost = (distance * time_per_meter) + (weight * time_per_ton)`

*   **Unloading Cost:** A simulated cost calculated on a completed plan. It includes:
    *   **Base Unload Cost:** The ideal time to unload every container.
    *   **Shifting Penalty:** A large penalty for every container that directly blocks another, calculated as `(ShiftCost * shifting_penalty_multiplier)`.

### 4. Input & Output

**Input File Format (input.json):** The problem is defined in a structured JSON file.
```json
{
    "containers_manifest": [
        { "id": "C001", "weight": 18, "destination": 2, "yard_coords": },
        { "id": "C002", "weight": 8,  "destination": 1, "yard_coords": }
    ],
    "ship_specification": {
        "bays": 3, "rows": 3, "tiers": 2,
        "max_com_deviation_radius": 0.8,
        "unloading_access_point":
    },
    "operational_costs": {
        "loading_time_per_meter": 0.05,
        "loading_time_per_ton": 0.2,
        "unloading_time_per_ton": 0.25,
        "shifting_penalty_multiplier": 2.0
    }
}
```

**Example Output (Console and output.txt):** The program produces a detailed report for each algorithm.
```shell
################## FINAL COMPARISON ##################
==================== A* Search Report ====================
Solution Found: YES
Total Lifecycle Cost: 45.98 minutes
  - Loading Cost:     16.98 minutes
  - Unloading Cost:   29.00 minutes
Time to Solve:        0.1490 seconds
Nodes/States Metric:  278 (nodes expanded)
========================================================
```

### 5. How to Compile and Run

The solution is implemented in Python.

1.  **Install Dependencies:**
    ```shell
    pip install numpy
    ```
2.  **Run the Program:**
    ```shell
    python final_solver.py
    ```
    *   Ensure `input.json` is in the same directory.
    *   Results will be displayed in the terminal and saved to `output.txt`.

### 6. Code Walkthrough & Heuristics

The code is structured into clear classes: `Container`, `State`, `ContainerLoadingEnvironment`, and solvers for `A*` and `Beam Search`.

The success of the solution relies on a powerful, two-part **Shared Heuristic** strategy:

1.  **Container Selection:** Always load the available container with the **farthest destination** first. This is the most effective way to minimize future unloading penalties.
2.  **Placement Scoring:** Intelligently score each valid placement based on how low it is in a stack and how well its distance from the gate matches its destination.

### 7. Analysis of Results

An experiment was conducted on a simplified 5-container instance to ensure both algorithms could run to completion. The results are definitive.

#### **Final Comparison Table**

| Algorithm | Solution Found | Total Cost (min) | Nodes/States Metric | Time (s) | Why this result? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A\* Search** | Yes | 45.98 | 278 (nodes expanded) | 0.1490 | To guarantee it found the absolute best solution, A\* methodically explored every single promising path. This exhaustive process is thorough but computationally intensive. |
| **Beam Search (k=20)** | Yes | 45.98 | 414 (states evaluated) | 0.0249 | It was guided directly to the best solution by its powerful internal heuristics. By intelligently pruning less-promising paths, it found the same optimal answer while being **6 times faster**. |

*   **Solution Quality:** Both algorithms found the exact same optimal loading plan. This highlights the exceptional quality of the heuristics used.
*   **Performance:** The performance difference is staggering. Beam Search was significantly faster because it intelligently pruned the vast majority of unpromising search paths.
*   **The Scalability Problem:** Most importantly, the fact that A\* failed completely on larger 8 and 10-container instances due to memory exhaustion proves its practical limitations. Beam Search, however, continued to produce solutions efficiently.

### 8. Conclusion

This project successfully demonstrated the power of applying AI search techniques to a complex logistics problem. The key takeaways are:

*   A **comprehensive problem model**, including 3D physics and a full lifecycle cost function, is essential for finding meaningful solutions.
*   While **A\*** serves as an essential theoretical benchmark, its exponential complexity makes it impractical for real-world problem sizes.
*   **Beam Search**, when guided by strong, domain-specific heuristics, offers the best of both worlds. It provides the **efficiency** to solve large problems quickly and the **intelligence** to find high-quality, near-optimal solutions.

Ultimately, this work shows that the best balance is struck by **Beam Search**, which ensures both effective performance and excellent, reliable results.
