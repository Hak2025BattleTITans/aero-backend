# overbooking_optimizer.py - файл алгоритма оптимизации овербукинга
import pandas as pd
import time
import numpy as np
import warnings
import random
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model  # pyright: ignore[reportMissingImports]
from tensorflow.keras.losses import mse  # pyright: ignore[reportMissingImports]
from optimizer.cores import overbooking_core
import os
import tempfile

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

def predict_with_model(model, df_requests, scaler_X, scaler_y, label_encoders, aircraft_capacities):
    input_columns = scaler_X.feature_names_in_
    if df_requests.empty: return np.array([])
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
    y_pred_scaled = model.predict(X_new_seq, batch_size=2048, verbose=0)
    y_pred_flat = y_pred_scaled.reshape(-1, y_pred_scaled.shape[-1])
    y_pred_original = scaler_y.inverse_transform(y_pred_flat)
    predictions_3d = y_pred_original.reshape(num_flights, 3, -1)
    final_predictions = np.zeros_like(predictions_3d)
    cabin_map = {'C': 0, 'W': 1, 'Y': 2}
    for i in range(num_flights):
        ac_type = df_requests.iloc[i]['Тип ВС']
        real_capacities = aircraft_capacities.get(ac_type)
        if real_capacities:
            flight_preds = predictions_3d[i].copy(); flight_preds[flight_preds < 0] = 0; flight_preds[:, 1] = np.clip(flight_preds[:, 1], 0, 1)
            reliable_lf = flight_preds[cabin_map['C'], 1]
            c_pax, c_rev = flight_preds[cabin_map['C'], 4], flight_preds[cabin_map['C'], 3]
            avg_rev_per_pax_c = c_rev / c_pax if c_pax > 0.5 else 5000
            for cabin_code, cabin_idx in cabin_map.items():
                if cabin_code in real_capacities:
                    real_capacity = real_capacities[cabin_code]
                    predicted_lf, rev_multiplier = reliable_lf, (1.0 if cabin_code == 'C' else 0.7 if cabin_code == 'W' else 0.4)
                    new_pax = int(real_capacity * predicted_lf); new_rev = new_pax * avg_rev_per_pax_c * rev_multiplier
                    final_predictions[i, cabin_idx, :] = [real_capacity, predicted_lf, new_pax, new_rev, new_pax]
                else: final_predictions[i, cabin_idx, :] = 0
        else:
            flight_preds = predictions_3d[i].copy(); flight_preds[flight_preds < 0] = 0; flight_preds[:, 1] = np.clip(flight_preds[:, 1], 0, 1); flight_preds[flight_preds[:, 4] < 0.5, 3] = 0.0
            final_predictions[i] = flight_preds
    return final_predictions.reshape(num_flights, 3, -1)

def run_overbooking_optimization(input_df):
    DATA_FILE = 'hackathon_data_main_with_numbers.csv'
    MODEL_FILE = '/app/src/optimizer/cores/cnn_lstm_flight_model.h5'

    SEARCH_WINDOWS = [30, 40]
    TIME_STEP_MINUTES = 5
    MAX_FLIGHTS_IN_SLOT = 3
    MIN_INTERVAL_MINUTES = 5
    LF_THRESHOLD = 1.0
    MIN_SIBLING_INTERVAL_MINUTES = 25

    INTERMEDIATE_OUTPUT_FILE = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w', encoding='utf-8-sig') as tmp:
            INTERMEDIATE_OUTPUT_FILE = tmp.name

        print("--- ЭТАП 1: ПЕРВИЧНАЯ ОПТИМИЗАЦИЯ ---")
        print("1.1. Загрузка компонентов модели и данных...")
        aircraft_capacities = get_aircraft_capacities(DATA_FILE)
        model = load_model(MODEL_FILE, custom_objects={'mse': mse})
        MODEL_OUTPUT_SHAPE = model.output_shape[-1]
        scaler_X, scaler_y, label_encoders = recreate_scalers_and_encoders(DATA_FILE, MODEL_OUTPUT_SHAPE)

        print(f"\n1.2. Поиск рейсов с LF >= {LF_THRESHOLD} в данных...")
        df_opt = input_df.copy()
        for col in ['LF Кабина', 'Емкость кабины', 'Доход пасс', 'Пассажиры']:
            if col in df_opt.columns: df_opt[col] = pd.to_numeric(df_opt[col].astype(str).str.replace(',', '.'), errors='coerce')
        df_opt['flight_group_id'] = df_opt['№'].astype(str).str.split('-').str[0]
        target_flights = [{'flight_group_id': group_id, 'datetime': pd.to_datetime(f"{group.iloc[0]['Дата вылета']} {group.iloc[0]['Время вылета']}"), 'flight_no': str(group.iloc[0]['Номер рейса']), 'dep_airport': group.iloc[0]['Аэропорт вылета'], 'arr_airport': group.iloc[0]['Аэропорт прилета'], 'aircraft_type': group.iloc[0]['Тип ВС']} for group_id, group in df_opt.groupby('flight_group_id') if group['LF Кабина'].max() >= LF_THRESHOLD]
        print(f"  Найдено {len(target_flights)} перегруженных рейсов для добавления дублирующих.")

        background_traffic = defaultdict(list)
        df_opt['datetime'] = pd.to_datetime(df_opt['Дата вылета'] + ' ' + df_opt['Время вылета'])
        for _, row in df_opt.iterrows():
            try:
                dep_dt = row['datetime']; duration = pd.to_datetime(row['Время прилета'], format='%H:%M') - pd.to_datetime(row['Время вылета'], format='%H:%M')
                if duration.total_seconds() < 0: duration += pd.Timedelta(days=1)
                arr_dt = dep_dt + duration
                background_traffic[(row['Аэропорт вылета'], dep_dt.date())].append(dep_dt)
                background_traffic[(row['Аэропорт прилета'], arr_dt.date())].append(arr_dt)
            except (ValueError, TypeError): continue

        print("\n1.3. Поиск слотов (оптимизированный пакетный режим)...")
        start_batch_time = time.time()

        all_candidate_requests, request_metadata = [], []
        max_window = max(SEARCH_WINDOWS)
        for flight in target_flights:
            num_steps = (max_window * 2) // TIME_STEP_MINUTES
            for step in range(num_steps + 1):
                offset_minutes = -max_window + step * TIME_STEP_MINUTES
                new_dt = flight['datetime'] + pd.Timedelta(minutes=offset_minutes)
                if abs((new_dt - flight['datetime']).total_seconds()) < MIN_SIBLING_INTERVAL_MINUTES * 60 or new_dt.date() != flight['datetime'].date(): continue
                request = {'Дата вылета': new_dt.strftime('%Y-%m-%d'), 'Номер рейса': int(flight['flight_no']), 'Аэропорт вылета': flight['dep_airport'], 'Аэропорт прилета': flight['arr_airport'], 'Время вылета': new_dt.strftime('%H:%M'), 'Время прилета': (new_dt + pd.Timedelta(minutes=120)).strftime('%H:%M'), 'Тип ВС': flight['aircraft_type']}
                all_candidate_requests.append(request)
                request_metadata.append({'group_id': flight['flight_group_id'], 'datetime': new_dt})

        print(f"  - Выполняется предсказание для {len(all_candidate_requests)} кандидатов...")
        df_all_candidates = pd.DataFrame(all_candidate_requests)
        all_predictions = predict_with_model(model, df_all_candidates, scaler_X, scaler_y, label_encoders, aircraft_capacities)
        all_profits = np.sum(all_predictions[:, :, 3], axis=1)

        grouped_candidates = defaultdict(list)
        for i, meta in enumerate(request_metadata):
            grouped_candidates[meta['group_id']].append({'profit': all_profits[i], 'datetime': meta['datetime'], 'prediction_data': all_predictions[i], 'request_data': all_candidate_requests[i]})

        new_flights_info, repredicted_flights_ids = {}, set()
        for flight in target_flights:
            group_id = flight['flight_group_id']
            candidates_for_flight = sorted(grouped_candidates.get(group_id, []), key=lambda x: x['profit'], reverse=True)
            if not candidates_for_flight: print(f"  -> Для рейса {flight['flight_no']} не найдено подходящих кандидатов."); continue

            candidate_options_for_cython = [(c['profit'], c['datetime'], i) for i, c in enumerate(candidates_for_flight)]
            duration = pd.Timedelta(minutes=120)
            found_idx_local = overbooking_core.find_best_slot(candidate_options_for_cython, dict(background_traffic), flight['dep_airport'], flight['arr_airport'], duration, MAX_FLIGHTS_IN_SLOT, MIN_INTERVAL_MINUTES)

            if found_idx_local != -1:
                best_candidate = candidates_for_flight[found_idx_local]
                new_flights_info[group_id] = {'prediction_data': best_candidate['prediction_data'], 'request_data': best_candidate['request_data']}
                repredicted_flights_ids.add(group_id)
            else: print(f"  -> Для рейса {flight['flight_no']} свободных слотов не найдено.")

        batch_time = time.time() - start_batch_time
        print(f"  Поиск слотов завершен за {batch_time:.2f} сек.")

        print(f"\n1.4. Пересчет {len(repredicted_flights_ids)} исходных рейсов и их дубликатов...")
        final_data_list = []
        for _, row in df_opt.iterrows():
            row_dict = row.to_dict()
            group_id = row['flight_group_id']
            if group_id in repredicted_flights_ids:
                row_dict['Изменения'] = 'Изменено к-во людей'
                new_flight_info = new_flights_info.get(group_id)
                if new_flight_info:
                    new_flight_lf = new_flight_info['prediction_data'][0, 1]
                    total_lf = row['LF Кабина'] + new_flight_lf
                    base_lf = total_lf / 2.0; noise = random.uniform(-0.03, 0.03)
                    lf_for_old_flight = min(base_lf + noise, 1.0); lf_for_new_flight = min(base_lf - noise, 1.0)
                    avg_revenue_per_pax = row['Доход пасс'] / row['Пассажиры'] if row['Пассажиры'] > 0 else 0
                    new_passengers_old = int(row['Емкость кабины'] * lf_for_old_flight); new_revenue_old = new_passengers_old * avg_revenue_per_pax
                    row_dict.update({'LF Кабина': lf_for_old_flight, 'Пассажиры': new_passengers_old, 'Бронирования': new_passengers_old, 'Доход пасс': new_revenue_old})
                    new_flights_info[group_id]['final_lf'] = lf_for_new_flight
            final_data_list.append(row_dict)

        for group_id, new_flight_info in new_flights_info.items():
            req_data, pred_data, final_lf_for_new = new_flight_info['request_data'], new_flight_info['prediction_data'], new_flight_info['final_lf']
            dt = pd.to_datetime(f"{req_data['Дата вылета']} {req_data['Время вылета']}"); duration = pd.Timedelta(hours=2); new_flight_group_id = f"{group_id}-N"
            for cabin_idx, cabin_code in enumerate(['C', 'W', 'Y']):
                if aircraft_capacities.get(req_data['Тип ВС'], {}).get(cabin_code):
                    capacity = int(pred_data[cabin_idx, 0]); new_passengers = int(capacity * final_lf_for_new)
                    avg_revenue_per_pax_new = pred_data[cabin_idx, 3] / pred_data[cabin_idx, 4] if pred_data[cabin_idx, 4] > 0 else 0
                    new_revenue = new_passengers * avg_revenue_per_pax_new
                    final_data_list.append({'№': f"{new_flight_group_id}-{cabin_code}", 'Изменения': 'Новый', 'Дата вылета': req_data['Дата вылета'], 'Номер рейса': req_data['Номер рейса'], 'Аэропорт вылета': req_data['Аэропорт вылета'], 'Аэропорт прилета': req_data['Аэропорт прилета'], 'Время вылета': dt.strftime('%H:%M'), 'Время прилета': (dt + duration).strftime('%H:%M'), 'Емкость кабины': capacity, 'LF Кабина': final_lf_for_new, 'Бронирования': new_passengers, 'Тип ВС': req_data['Тип ВС'], 'Код кабины': cabin_code, 'Доход пасс': new_revenue, 'Пассажиры': new_passengers})

        df_intermediate = pd.DataFrame(final_data_list)
        if 'datetime' in df_intermediate.columns: df_intermediate = df_intermediate.drop(columns=['datetime'])
        if 'flight_group_id' in df_intermediate.columns: df_intermediate = df_intermediate.drop(columns=['flight_group_id'])
        df_intermediate.to_csv(INTERMEDIATE_OUTPUT_FILE, index=False, sep=';', encoding='utf-8-sig')
        print(f"\n1.5. Сборка промежуточного расписания завершена.")
        print("--- ЭТАП 1 ЗАВЕРШЕН ---\n")

        print("--- ЭТАП 2: ВТОРИЧНАЯ ОПТИМИЗАЦИЯ ---")
        print(f"2.1. Поиск рейсов с LF >= {LF_THRESHOLD} в промежуточном файле...")
        df_pass1 = pd.read_csv(INTERMEDIATE_OUTPUT_FILE, delimiter=';', encoding='utf-8-sig')
        for col in ['LF Кабина', 'Емкость кабины', 'Доход пасс', 'Пассажиры']:
            if col in df_pass1.columns: df_pass1[col] = pd.to_numeric(df_pass1[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
        df_pass1['base_group_id'] = df_pass1['№'].astype(str).str.split('-').str[0]

        target_groups_pass2 = {}
        grouped = df_pass1.groupby('base_group_id')
        for group_id, group in grouped:
            if group['LF Кабина'].max() >= LF_THRESHOLD:
                original_flight = group[~group['№'].str.contains('-N')].iloc[0]
                dt_str = f"{original_flight['Дата вылета']} {original_flight['Время вылета']}"
                existing_times_str = (group['Дата вылета'] + ' ' + group['Время вылета']).unique()
                existing_times = [pd.to_datetime(ts) for ts in existing_times_str]
                target_groups_pass2[group_id] = {'original_datetime': pd.to_datetime(dt_str), 'flight_no': str(original_flight['Номер рейса']), 'dep_airport': original_flight['Аэропорт вылета'], 'arr_airport': original_flight['Аэропорт прилета'], 'aircraft_type': original_flight['Тип ВС'], 'existing_datetimes': set(existing_times), 'existing_flights_count': len(group['№'].str.replace(r'-[C|W|Y]$', '', regex=True).unique())}
        print(f"  Найдено {len(target_groups_pass2)} групп рейсов для добавления еще одного дубликата.")

        if not target_groups_pass2:
            print("  Рейсов для второго этапа оптимизации не найдено. Завершение.")
            return df_pass1
        else:
            background_traffic_pass2 = defaultdict(list)
            for _, row in df_pass1.iterrows():
                try:
                    dep_dt = pd.to_datetime(f"{row['Дата вылета']} {row['Время вылета']}"); duration = pd.to_datetime(row['Время прилета'], format='%H:%M') - pd.to_datetime(row['Время вылета'], format='%H:%M')
                    if duration.total_seconds() < 0: duration += pd.Timedelta(days=1)
                    arr_dt = dep_dt + duration
                    background_traffic_pass2[(row['Аэропорт вылета'], dep_dt.date())].append(dep_dt)
                    background_traffic_pass2[(row['Аэропорт прилета'], arr_dt.date())].append(arr_dt)
                except (ValueError, TypeError): continue

            new_flights_info_pass2 = {}
            print("\n2.2. Итеративный поиск слотов для дополнительных рейсов...")
            for group_id, flight_info in target_groups_pass2.items():
                print(f"  - Поиск для группы {group_id}...")
                window_size = 30
                while window_size <= 180:
                    print(f"    -> Поиск в окне +/- {window_size} минут...")
                    candidate_requests = []
                    num_steps = (window_size * 2) // TIME_STEP_MINUTES
                    for step in range(num_steps + 1):
                        offset = -window_size + step * TIME_STEP_MINUTES
                        new_dt = flight_info['original_datetime'] + pd.Timedelta(minutes=offset)
                        is_too_close = any(abs((new_dt - existing_dt).total_seconds()) < MIN_SIBLING_INTERVAL_MINUTES * 60 for existing_dt in flight_info['existing_datetimes'])
                        if is_too_close or new_dt.date() != flight_info['original_datetime'].date(): continue
                        candidate_requests.append({'Дата вылета': new_dt.strftime('%Y-%m-%d'), 'Номер рейса': int(flight_info['flight_no']), 'Аэропорт вылета': flight_info['dep_airport'], 'Аэропорт прилета': flight_info['arr_airport'], 'Время вылета': new_dt.strftime('%H:%M'), 'Время прилета': (new_dt + pd.Timedelta(minutes=120)).strftime('%H:%M'), 'Тип ВС': flight_info['aircraft_type']})
                    if not candidate_requests: window_size += 10; continue
                    df_candidates = pd.DataFrame(candidate_requests)
                    candidate_predictions = predict_with_model(model, df_candidates, scaler_X, scaler_y, label_encoders, aircraft_capacities)
                    if candidate_predictions.size == 0: window_size += 10; continue
                    candidate_options = [(profit, pd.to_datetime(f"{df_candidates.iloc[j]['Дата вылета']} {df_candidates.iloc[j]['Время вылета']}"), j) for j, profit in enumerate(np.sum(candidate_predictions[:, :, 3], axis=1))]
                    duration = pd.Timedelta(minutes=120)
                    found_idx_local = overbooking_core.find_best_slot(candidate_options, dict(background_traffic_pass2), flight_info['dep_airport'], flight_info['arr_airport'], duration, MAX_FLIGHTS_IN_SLOT, MIN_INTERVAL_MINUTES)
                    if found_idx_local != -1:
                        original_request_index = candidate_options[found_idx_local][2]
                        group_flights = df_pass1[df_pass1['base_group_id'] == group_id]
                        total_existing_lf = group_flights.groupby(group_flights['№'].str.replace(r'-[C|W|Y]$', '', regex=True))['LF Кабина'].mean().sum()
                        predicted_lf_new = candidate_predictions[original_request_index][0, 1]
                        avg_lf = (total_existing_lf + predicted_lf_new) / (flight_info['existing_flights_count'] + 1)
                        if avg_lf < 1.0:
                            print(f"      -> Слот найден! Средний LF < 1.0. Фиксируем.")
                            extracted_suffixes_df = group_flights['№'].str.extract(r'(-N\d*)-')
                            new_flight_count = len(extracted_suffixes_df[0].dropna().unique())
                            new_flight_suffix = f"N{new_flight_count}" if new_flight_count > 0 else "N"
                            new_flights_info_pass2[group_id] = {'prediction_data': candidate_predictions[original_request_index], 'request_data': candidate_requests[original_request_index], 'avg_lf': avg_lf, 'new_id_suffix': new_flight_suffix}
                            break
                        else: print(f"      -> Слот найден, но средний LF ({avg_lf:.2f}) все еще >= 1.0. Расширяем поиск...")
                    else: print(f"      -> Свободных слотов не найдено.")
                    window_size += 10
                if group_id not in new_flights_info_pass2: print(f"  -> Итого: для группы {group_id} подходящих слотов не найдено.")

            print(f"\n2.3. Финальная сборка расписания с {len(new_flights_info_pass2)} дополнительными рейсами...")
            final_rows = []
            for _, row in df_pass1.iterrows():
                row_dict = row.to_dict()
                base_id = row['base_group_id']
                if base_id in new_flights_info_pass2:
                    avg_lf = new_flights_info_pass2[base_id]['avg_lf']; noise = random.uniform(-0.02, 0.02); final_lf = min(avg_lf + noise, 0.99)
                    avg_revenue_per_pax = row['Доход пасс'] / row['Пассажиры'] if row['Пассажиры'] > 0 else 0
                    new_passengers = int(row['Емкость кабины'] * final_lf); new_revenue = new_passengers * avg_revenue_per_pax
                    row_dict.update({'LF Кабина': final_lf, 'Пассажиры': new_passengers, 'Бронирования': new_passengers, 'Доход пасс': new_revenue, 'Изменения': 'Изменено к-во людей'})
                final_rows.append(row_dict)

            for group_id, info in new_flights_info_pass2.items():
                req_data, pred_data, avg_lf = info['request_data'], info['prediction_data'], info['avg_lf']
                final_lf_for_new = min(avg_lf + random.uniform(-0.02, 0.02), 0.99)
                dt = pd.to_datetime(f"{req_data['Дата вылета']} {req_data['Время вылета']}"); duration = pd.Timedelta(hours=2); new_flight_group_id = f"{group_id}-{info['new_id_suffix']}"
                for cabin_idx, cabin_code in enumerate(['C', 'W', 'Y']):
                    if aircraft_capacities.get(req_data['Тип ВС'], {}).get(cabin_code):
                        capacity = int(pred_data[cabin_idx, 0]); new_passengers = int(capacity * final_lf_for_new)
                        avg_revenue_per_pax_new = pred_data[cabin_idx, 3] / pred_data[cabin_idx, 4] if pred_data[cabin_idx, 4] > 0 else 0
                        new_revenue = new_passengers * avg_revenue_per_pax_new
                        final_rows.append({'№': f"{new_flight_group_id}-{cabin_code}", 'Изменения': 'Новый', 'Дата вылета': req_data['Дата вылета'], 'Номер рейса': req_data['Номер рейса'], 'Аэропорт вылета': req_data['Аэропорт вылета'], 'Аэропорт прилета': req_data['Аэропорт прилета'], 'Время вылета': dt.strftime('%H:%M'), 'Время прилета': (dt + duration).strftime('%H:%M'), 'Емкость кабины': capacity, 'LF Кабина': final_lf_for_new, 'Бронирования': new_passengers, 'Тип ВС': req_data['Тип ВС'], 'Код кабины': cabin_code, 'Доход пасс': new_revenue, 'Пассажиры': new_passengers})

            df_final = pd.DataFrame(final_rows)
            if 'base_group_id' in df_final.columns: df_final = df_final.drop(columns=['base_group_id'])
            df_final.sort_values(by=['Дата вылета', 'Время вылета', 'Номер рейса'], inplace=True)
            print(f"  Итоговое полностью оптимизированное расписание готово")

        print("\n--- ОПТИМИЗАЦИЯ ПОЛНОСТЬЮ ЗАВЕРШЕНА ---")
        return df_final

    finally:
        if INTERMEDIATE_OUTPUT_FILE and os.path.exists(INTERMEDIATE_OUTPUT_FILE):
            os.remove(INTERMEDIATE_OUTPUT_FILE)

if __name__ == "__main__":
    start_time = time.time()
    default_input_file = 'final_schedule_optimized.csv'
    default_output_file = 'final_schedule_fully_optimized.csv'

    print(f"Запуск оптимизатора для файла: {default_input_file}")
    final_file_path = run_overbooking_optimization(
        initial_schedule_file=default_input_file,
        final_output_file=default_output_file
    )
    if final_file_path:
        print(f"\nРабота завершена. Итоговый файл: {final_file_path}")
    else:
        print("\nРабота завершилась с ошибкой.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Общее время выполнения: {minutes} мин {seconds} сек.")