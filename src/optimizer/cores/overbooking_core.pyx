# overbooking_core.pyx - алгоритм cython для оптимизации овербукинга, проверка временного слота по аэропортам
import cython
cimport cython

def find_best_slot(
    list candidate_options,
    dict background_traffic,
    str dep_airport,
    str arr_airport,
    object duration,
    int max_flights_in_slot,
    int min_interval_minutes
):
    cdef int interval_seconds = min_interval_minutes * 60
    cdef int conflict_count
    cdef bint is_dep_slot_free, is_arr_slot_free

    candidate_options.sort(key=lambda x: x[0], reverse=True)
    
    for profit, dep_dt, original_index in candidate_options:
        arr_dt = dep_dt + duration
        
        is_dep_slot_free = True
        dep_key = (dep_airport, dep_dt.date())
        if dep_key in background_traffic:
            conflict_count = 0
            for event_time in background_traffic[dep_key]:
                if abs((event_time - dep_dt).total_seconds()) < interval_seconds:
                    conflict_count += 1
            if conflict_count >= max_flights_in_slot:
                is_dep_slot_free = False
        
        if not is_dep_slot_free:
            continue

        is_arr_slot_free = True
        arr_key = (arr_airport, arr_dt.date())
        if arr_key in background_traffic:
            conflict_count = 0
            for event_time in background_traffic[arr_key]:
                if abs((event_time - arr_dt).total_seconds()) < interval_seconds:
                    conflict_count += 1
            if conflict_count >= max_flights_in_slot:
                is_arr_slot_free = False
        
        if is_arr_slot_free:
            return original_index
            
    return -1