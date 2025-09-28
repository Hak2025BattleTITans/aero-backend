# main_ranking.py - файл алгоритма оптимизации перемещением
import pandas as pd
import time
import numpy as np
import warnings
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mse
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix, vstack
from optimizer.cores import optimizer_core
import os

warnings.filterwarnings("ignore")

def get_aircraft_capacities(data_path):
    df = pd.read_csv(data_path, delimiter=';', encoding='utf-8-sig')
    df['Емкость кабины'] = pd.to_numeric(df['Емкость кабины'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
    capacities = {}
    for _, row in df.iterrows():
        ac_type, cabin, capacity = row['Тип ВС'], row['Код кабины'], int(row['Емкость кабины'])
        if ac_type not in capacities: capacities[ac_type] = {}
        if capacity > 0:
            if cabin not in capacities[ac_type] or capacity > capacities[ac_type][cabin]:
                capacities[ac_type][cabin] = capacity
    return capacities

def predict_with_model(model, df_requests, scaler_X, scaler_y, label_encoders, aircraft_capacities):
    input_columns = scaler_X.feature_names_in_
    num_flights = len(df_requests)
    df_for_pred = pd.concat([df_requests.assign(**{'Код кабины': c}) for c in ['C', 'W', 'Y']]).sort_index()
    new_data_encoded = df_for_pred.copy()
    for col, le in label_encoders.items():
        if col in new_data_encoded.columns:
            known_mask = new_data_encoded[col].isin(le.classes_)
            new_data_encoded.loc[known_mask, col] = le.transform(new_data_encoded.loc[known_mask, col])
            new_data_encoded.loc[~known_mask, col] = -1
    new_data_encoded['Дата вылета'] = pd.to_datetime(new_data_encoded['Дата вылета']).astype('int64') // 10**9
    new_data_encoded['Время вылета'] = pd.to_datetime(new_data_encoded['Время вылета'], format='%H:%M').dt.hour * 3600 + pd.to_datetime(new_data_encoded['Время вылета'], format='%H:%M').dt.minute * 60
    new_data_encoded['Время прилета'] = pd.to_datetime(new_data_encoded['Время прилета'], format='%H:%M').dt.hour * 3600 + pd.to_datetime(new_data_encoded['Время прилета'], format='%H:%M').dt.minute * 60
    for col in input_columns:
        if col not in new_data_encoded.columns: new_data_encoded[col] = 0
    X_new = scaler_X.transform(new_data_encoded[input_columns].astype(float))
    X_new_seq = X_new.reshape(num_flights, 3, -1)
    y_pred_scaled = model.predict(X_new_seq, batch_size=1024, verbose=0)
    y_pred_flat = y_pred_scaled.reshape(-1, y_pred_scaled.shape[-1])
    y_pred_original = scaler_y.inverse_transform(y_pred_flat)
    predictions_3d = y_pred_original.reshape(num_flights, 3, -1)
    final_predictions = np.zeros_like(predictions_3d)
    cabin_map = {'C': 0, 'W': 1, 'Y': 2}
    for i in range(num_flights):
        ac_type = df_requests.iloc[i]['Тип ВС']
        real_capacities = aircraft_capacities.get(ac_type)
        if real_capacities:
            flight_preds = predictions_3d[i].copy()
            flight_preds[flight_preds < 0] = 0
            flight_preds[:, 1] = np.clip(flight_preds[:, 1], 0, 1)
            reliable_lf = flight_preds[cabin_map['C'], 1]
            c_pax, c_rev = flight_preds[cabin_map['C'], 4], flight_preds[cabin_map['C'], 3]
            avg_rev_per_pax_c = c_rev / c_pax if c_pax > 0.5 else 5000
            for cabin_code, cabin_idx in cabin_map.items():
                if cabin_code in real_capacities:
                    real_capacity = real_capacities[cabin_code]
                    predicted_lf, rev_multiplier = reliable_lf, (1.0 if cabin_code == 'C' else 0.7 if cabin_code == 'W' else 0.4)
                    new_pax = int(real_capacity * predicted_lf)
                    new_rev = new_pax * avg_rev_per_pax_c * rev_multiplier
                    final_predictions[i, cabin_idx, :] = [real_capacity, predicted_lf, new_pax, new_rev, new_pax]
                else: final_predictions[i, cabin_idx, :] = 0
        else:
            flight_preds = predictions_3d[i].copy()
            flight_preds[flight_preds < 0] = 0
            flight_preds[:, 1] = np.clip(flight_preds[:, 1], 0, 1)
            flight_preds[flight_preds[:, 4] < 0.5, 3] = 0.0
            final_predictions[i] = flight_preds
    return final_predictions.reshape(num_flights, 3, -1)
def recreate_scalers_and_encoders(data_path, model_output_shape):
    df = pd.read_csv(data_path, delimiter=';', encoding='utf-8-sig')
    categorical_cols = ['Аэропорт вылета', 'Аэропорт прилета', 'Тип ВС', 'Код кабины', 'Номер рейса']
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns: le = LabelEncoder(); df[col] = le.fit_transform(df[col].astype(str)); label_encoders[col] = le
    df['Дата вылета'] = pd.to_datetime(df['Дата вылета']).astype('int64') // 10**9
    df['Время вылета'] = pd.to_datetime(df['Время вылета'], format='%H:%M', errors='coerce').dt.hour * 3600 + pd.to_datetime(df['Время вылета'], format='%H:%M', errors='coerce').dt.minute * 60
    df['Время прилета'] = pd.to_datetime(df['Время прилета'], format='%H:%M', errors='coerce').dt.hour * 3600 + pd.to_datetime(df['Время прилета'], format='%H:%M', errors='coerce').dt.minute * 60
    df.fillna(0, inplace=True)
    input_cols = ['Дата вылета', 'Номер рейса', 'Аэропорт вылета', 'Аэропорт прилета', 'Время вылета', 'Время прилета', 'Тип ВС', 'Код кабины']
    output_cols = ['Емкость кабины', 'LF Кабина', 'Бронирования по кабинам', 'Доход пасс', 'Пассажиры']
    scaler_X = StandardScaler().fit(df[input_cols])
    for col in output_cols: df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    df.fillna(0, inplace=True)
    small_scaler_y = StandardScaler().fit(df[output_cols])
    scaler_y = StandardScaler(); scaler_y.n_features_in_ = model_output_shape
    mean_ = np.zeros(model_output_shape); scale_ = np.ones(model_output_shape)
    mean_[:len(output_cols)] = small_scaler_y.mean_; scale_[:len(output_cols)] = small_scaler_y.scale_
    scaler_y.mean_ = mean_; scaler_y.scale_ = scale_
    return scaler_X, scaler_y, label_encoders

def load_and_prepare_data(data_path):
    df_schedule = pd.read_csv(data_path, delimiter=';', encoding='utf-8-sig')
    df_schedule['flight_group_id'] = df_schedule['№'].astype(str).str.split('-').str[0]
    for col in ['Емкость кабины', 'LF Кабина', 'Бронирования по кабинам', 'Доход пасс', 'Пассажиры']:
        if col in df_schedule.columns:
            df_schedule[col] = pd.to_numeric(df_schedule[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
    original_data_map = { group_id: group.to_dict('records') for group_id, group in df_schedule.groupby('flight_group_id') }
    aggregated_flights = []
    for group_id, group in df_schedule.groupby('flight_group_id'):
        first_row = group.iloc[0]
        try:
            dep_time = pd.to_datetime(f"{first_row['Дата вылета']} {first_row['Время вылета']}", format='%Y-%m-%d %H:%M')
            arr_time = pd.to_datetime(f"{first_row['Дата вылета']} {first_row['Время прилета']}", format='%Y-%m-%d %H:%M')
        except (ValueError, TypeError): continue
        if pd.isna(dep_time) or pd.isna(arr_time): continue
        if arr_time < dep_time: arr_time += pd.Timedelta(days=1)
        duration = (arr_time - dep_time)
        aggregated_flights.append({
            'flight_group_id': group_id,
            'static_data': { 'flight_no': str(first_row['Номер рейса']), 'dep_airport': first_row['Аэропорт вылета'], 'arr_airport': first_row['Аэропорт прилета'], 'duration': duration, 'aircraft_type': first_row['Тип ВС'], 'initial_datetime': dep_time, 'available_cabins': sorted(group['Код кабины'].unique().tolist()) }
        })
    return aggregated_flights, original_data_map

def run_ranking_optimization(data_file, optimized_file):
    MODEL_FILE = '/app/src/optimizer/cores/cnn_lstm_flight_model.h5'
    TIME_WINDOW_HOURS, TIME_STEP_MINUTES, MIN_INTERVAL_MINUTES, MAX_FLIGHTS_IN_INTERVAL = 1, 15, 5, 2

    print("1. Загрузка данных и компонентов модели...");
    aircraft_capacities = get_aircraft_capacities(data_file)
    model = load_model(MODEL_FILE, custom_objects={'mse': mse})
    MODEL_OUTPUT_SHAPE = model.output_shape[-1]
    scaler_X, scaler_y, label_encoders = recreate_scalers_and_encoders(data_file, MODEL_OUTPUT_SHAPE)
    flights, original_data_map = load_and_prepare_data(data_file)
    print(f"Найдено {len(flights)} рейсов для оптимизации.")

    print("\n2. Создание 'Вселенной вариантов'...")
    decision_vars, all_requests_list = [], []
    for i, flight_data in enumerate(flights):
        flight = flight_data['static_data']
        initial_date = flight['initial_datetime'].date()
        for step in range(((TIME_WINDOW_HOURS * 2 * 60) // TIME_STEP_MINUTES) + 1):
            offset_minutes = -TIME_WINDOW_HOURS * 60 + step * TIME_STEP_MINUTES
            new_dt = flight['initial_datetime'] + pd.Timedelta(minutes=offset_minutes)
            if new_dt.date() != initial_date: continue
            decision_vars.append({'flight_index': i, 'datetime': new_dt})
            all_requests_list.append({'Дата вылета': new_dt.strftime('%Y-%m-%d'), 'Номер рейса': int(flight['flight_no']), 'Аэропорт вылета': flight['dep_airport'], 'Аэропорт прилета': flight['arr_airport'], 'Время вылета': new_dt.strftime('%H:%M'), 'Время прилета': (new_dt + flight['duration']).strftime('%H:%M'), 'Тип ВС': flight['aircraft_type']})
    num_vars = len(decision_vars)
    print(f"  Создано {num_vars} возможных решений (переменных).")

    print("\n3. Пакетный прогноз с помощью гибридной модели...");
    if not all_requests_list:
        print("  Нет возможных решений для оптимизации. Проверьте временное окно и данные.")
        return None
    df_requests = pd.DataFrame(all_requests_list)
    all_predictions = predict_with_model(model, df_requests, scaler_X, scaler_y, label_encoders, aircraft_capacities)

    print("\n4. Построение матрицы ограничений с помощью Cython...")
    start_time_matrix = time.time()
    profits = np.sum(all_predictions[:, :, 3], axis=1)
    c = -profits
    flight_data_arr = np.ones(num_vars, dtype=np.float64)
    flight_row_ind = np.array([dv['flight_index'] for dv in decision_vars], dtype=np.intp)
    flight_col_ind = np.arange(num_vars, dtype=np.intp)
    flight_constraints = coo_matrix((flight_data_arr, (flight_row_ind, flight_col_ind)), shape=(len(flights), num_vars))
    grouped_events = defaultdict(list)
    for i, var in enumerate(decision_vars):
        static = flights[var['flight_index']]['static_data']
        dep_dt = var['datetime']; arr_dt = dep_dt + static['duration']
        grouped_events[(static['dep_airport'], dep_dt.date())].append((dep_dt, i))
        grouped_events[(static['arr_airport'], arr_dt.date())].append((arr_dt, i))
    slot_triplets = optimizer_core.build_slot_constraints(dict(grouped_events), MIN_INTERVAL_MINUTES, MAX_FLIGHTS_IN_INTERVAL)
    if slot_triplets:
        slot_data, slot_row, slot_col = zip(*slot_triplets)
        num_slot_constraints = max(slot_row) + 1 if slot_row else 0
        slot_constraints = coo_matrix((slot_data, (slot_row, slot_col)), shape=(num_slot_constraints, num_vars))
        A = vstack([flight_constraints, slot_constraints], format='csr')
        lb, ub = np.concatenate([np.ones(len(flights)), np.zeros(num_slot_constraints)]), np.concatenate([np.ones(len(flights)), np.full(num_slot_constraints, MAX_FLIGHTS_IN_INTERVAL)])
    else:
        A, lb, ub = flight_constraints.tocsr(), np.ones(len(flights)), np.ones(len(flights))
    constraints = LinearConstraint(A, lb=lb, ub=ub)
    model_build_time = time.time() - start_time_matrix
    print(f"  Модель полностью построена за {model_build_time:.2f} сек.")

    print("\n5. Запуск решателя MILP...")
    start_solve_time = time.time()
    integrality, bounds = np.ones(num_vars), Bounds(lb=0, ub=1)
    result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options={'disp': True})
    solve_time = time.time() - start_solve_time
    print(f"  Решение найдено за {solve_time:.2f} сек.")

    if result.success:
        optimal_profit = -result.fun
        print("\n--- Общие результаты ---"); print(f"Оптимальный доход: {optimal_profit:,.2f} у.е.")
        print("\n6. Сохранение оптимального расписания...")
        final_schedule_data = []
        chosen_vars_indices = np.where(result.x > 0.9)[0]
        chosen_predictions = all_predictions[chosen_vars_indices]
        cabin_to_pred_idx = {'C': 0, 'W': 1, 'Y': 2}
        for i, var_idx in enumerate(chosen_vars_indices):
            var_info = decision_vars[var_idx]; flight = flights[var_info['flight_index']]; s = flight['static_data']; dt = var_info['datetime']
            change_status = "Изменено" if dt != s['initial_datetime'] else "Не изменено"
            if change_status == "Не изменено":
                for original_row in original_data_map.get(flight['flight_group_id'], []):
                    final_schedule_data.append({'№': original_row['№'], 'Изменения': change_status, 'Дата вылета': dt.strftime('%Y-%m-%d'), 'Номер рейса': s['flight_no'], 'Аэропорт вылета': s['dep_airport'], 'Аэропорт прилета': s['arr_airport'], 'Время вылета': dt.strftime('%H:%M'), 'Время прилета': (dt + s['duration']).strftime('%H:%M'), 'Емкость кабины': int(original_row['Емкость кабины']), 'LF Кабина': round(original_row['LF Кабина'], 4), 'Бронирования': int(original_row['Бронирования по кабинам']), 'Тип ВС': s['aircraft_type'], 'Код кабины': original_row['Код кабины'], 'Доход пасс': round(original_row['Доход пасс'], 2), 'Пассажиры': int(original_row['Пассажиры'])})
            else:
                preds_for_flight = chosen_predictions[i]
                for cabin_code in s['available_cabins']:
                    pred = preds_for_flight[cabin_to_pred_idx[cabin_code]]
                    final_schedule_data.append({'№': f"{flight['flight_group_id']}-{cabin_code}", 'Изменения': change_status, 'Дата вылета': dt.strftime('%Y-%m-%d'), 'Номер рейса': s['flight_no'], 'Аэропорт вылета': s['dep_airport'], 'Аэропорт прилета': s['arr_airport'], 'Время вылета': dt.strftime('%H:%M'), 'Время прилета': (dt + s['duration']).strftime('%H:%M'), 'Емкость кабины': int(pred[0]), 'LF Кабина': round(float(pred[1]), 4), 'Бронирования': int(pred[2]), 'Тип ВС': s['aircraft_type'], 'Код кабины': cabin_code, 'Доход пасс': round(float(pred[3]), 2), 'Пассажиры': int(pred[4])})
        df_final = pd.DataFrame(final_schedule_data)
        if not df_final.empty:
            df_final['sort_key'] = df_final['№'].apply(lambda x: (int(x.split('-')[0]), x.split('-')[1]))
            df_final = df_final.sort_values('sort_key').drop(columns='sort_key')
        df_final.to_csv(optimized_file, index=False, sep=';', encoding='utf-8-sig')
        print(f"  Оптимальное расписание сохранено в '{optimized_file}'")
        return optimized_file
    else:
        print(f"\nОШИБКА: Решателю не удалось найти решение. Статус: {result.message}")
        return None

if __name__ == "__main__":
    start_total_time = time.time()

    default_data_file = 'hackathon_data_main_with_numbers.csv'
    default_optimized_file = 'final_schedule_optimized.csv'
    default_model_file = 'cnn_lstm_flight_model.h5'

    print(f"Запуск оптимизатора ранжирования для файла: {default_data_file}")

    final_file_path = run_ranking_optimization(
        data_file=default_data_file,
        optimized_file=default_optimized_file
    )

    if final_file_path:
        print(f"\nРабота успешно завершена. Итоговый файл: {final_file_path}")
    else:
        print("\nРабота завершилась с ошибкой.")

    end_total_time = time.time()
    elapsed_time = end_total_time - start_total_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"\nОбщее время выполнения скрипта: {minutes} мин {seconds} сек.")