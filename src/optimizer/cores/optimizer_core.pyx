# optimizer_core.pyx - алгоритм cython для оптимизации перемещением, используя скользящий алгоритм для прохода по матрице
import cython
import numpy as np
cimport numpy as np

def build_slot_constraints(
    dict grouped_events,
    int min_interval_minutes,
    int max_flights_in_interval
):
    cdef list slot_constraints_triplets = []
    cdef int current_slot_idx = 0
    cdef int n_events, left_ptr, right_ptr
    
    for group_key, events_in_group in grouped_events.items():
        events_in_group.sort(key=lambda x: x[0])
        
        n_events = len(events_in_group)
        if n_events <= max_flights_in_interval:
            continue

        right_ptr = 0
        
        for left_ptr in range(n_events):
            while right_ptr < n_events and (events_in_group[right_ptr][0] - events_in_group[left_ptr][0]).total_seconds() < min_interval_minutes * 60:
                right_ptr += 1
            
            if (right_ptr - left_ptr) > max_flights_in_interval:
                constraint_vars = {ev[1] for ev in events_in_group[left_ptr:right_ptr]}
                for var_idx in constraint_vars:
                    slot_constraints_triplets.append((1, current_slot_idx, var_idx))
                current_slot_idx += 1

    return slot_constraints_triplets