# report_generator.py - файл создания эксель отчетов
import pandas as pd
import numpy as np

def create_comparison_report(file_info: list, output_file: str):
    """
    Создает Excel-отчет для сравнения метрик (общие итоги и по дням) из нескольких файлов.
    
    Args:
        file_info (list): Список кортежей, где каждый кортеж содержит (путь_к_файлу, метка_для_колонок).
                          Пример: [('file1.csv', 'До'), ('file2.csv', 'После')]
        output_file (str): Путь для сохранения итогового Excel-файла.
    """
    print(f"\n--- Создание отчета '{output_file}' ---")
    
    def preprocess_df(filepath):
        try:
            df = pd.read_csv(filepath, delimiter=';', encoding='utf-8-sig')
            rename_dict = {'Доход пасс': 'Доход', 'Пассажиры': 'Пассажиропоток'}
            df.rename(columns=rename_dict, inplace=True)
            for col in ['Доход', 'Пассажиропоток']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
            df['Дата вылета'] = pd.to_datetime(df['Дата вылета'], errors='coerce')
            return df
        except FileNotFoundError:
            print(f"  ОШИБКА: Не удалось найти файл: {filepath}")
            return None

    processed_dfs = []
    labels = []
    for path, label in file_info:
        df = preprocess_df(path)
        if df is None: return
        processed_dfs.append(df)
        labels.append(label)

    def get_summary_by_day():
        aggregated_dfs = []
        for i, df in enumerate(processed_dfs):
            df_grouped = df.copy()
            df_grouped['date'] = df_grouped['Дата вылета'].dt.date
            
            agg = df_grouped.groupby('date').agg({'Пассажиропоток': 'sum', 'Доход': 'sum'})
            agg['Средний чек'] = np.where(agg['Пассажиропоток'] > 0, agg['Доход'] / agg['Пассажиропоток'], 0)
            agg.columns = [f"{col}_{labels[i]}" for col in agg.columns]
            aggregated_dfs.append(agg)
        
        df_comp = aggregated_dfs[0]
        for i in range(1, len(aggregated_dfs)):
            df_comp = pd.merge(df_comp, aggregated_dfs[i], on='date', how='outer')
        
        df_comp.fillna(0, inplace=True)
        
        order = []
        for metric in ['Пассажиропоток', 'Доход', 'Средний чек']:
            order.extend([f"{metric}_{label}" for label in labels])
        return df_comp[order]

    df_by_day = get_summary_by_day()
    
    total_data = {}
    for i, df in enumerate(processed_dfs):
        total = df[['Пассажиропоток', 'Доход']].sum()
        total['Средний чек'] = (total['Доход'] / total['Пассажиропоток']) if total['Пассажиропоток'] > 0 else 0
        total_data[labels[i]] = total
    df_total = pd.DataFrame(total_data)
    
    def auto_adjust_xlsx_columns(df, writer, sheet_name):
        df_to_save = df.reset_index()
        df_to_save.to_excel(writer, sheet_name=sheet_name, index=False, float_format="%.2f")
        worksheet = writer.sheets[sheet_name]
        for idx, col in enumerate(df_to_save.columns):
            series = df_to_save[col]
            max_len = max((series.astype(str).map(len).max(), len(str(series.name)))) + 2
            worksheet.set_column(idx, idx, max_len)
    
    try:
        with pd.ExcelWriter(output_file, engine='xlsxwriter', datetime_format='yyyy-mm-dd') as writer:
            auto_adjust_xlsx_columns(df_total, writer, 'Итоги за весь период')
            auto_adjust_xlsx_columns(df_by_day, writer, 'Сравнение по дням')
        print(f"  Отчет успешно сохранен в '{output_file}'")
    except PermissionError:
        print(f"  ОШИБКА: Не удалось сохранить отчет. Файл '{output_file}' открыт. Закройте его и попробуйте снова.")
    except Exception as e:
        print(f"  Произошла непредвиденная ошибка при сохранении отчета: {e}")