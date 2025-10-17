import logging
import os
from logging.config import dictConfig
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd

from logging_config import LOGGING_CONFIG, ColoredFormatter
from .report_generator import create_comparison_report
from .main_ranking import run_ranking_optimization
from .overbooking_optimizer import run_overbooking_optimization

# ===== Paths setup ======
env_path = find_dotenv(".env", usecwd=True)
load_dotenv(env_path, override=True, encoding="utf-8-sig")

OPTIMIZED_DIR = Path(os.environ.get("OPTIMIZED_DIR", "optimized"))
OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = Path(os.environ.get("OPTIMIZED_DIR", "optimized"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Csv configurations
CONFIG_FILE = "/app/src/optimizer/configurators/configs.csv"
HISTORICAL_DATA_FILE = "/app/src/optimizer/configurators/historical.csv"

# ===== Logging setup ======
dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if type(handler) is logging.StreamHandler:
        handler.setFormatter(ColoredFormatter('%(levelname)s:     %(asctime)s %(name)s - %(message)s'))


def check_files_exist(filenames):
    """Проверяет, существуют ли все файлы в списке."""
    for filename in filenames:
        if not os.path.exists(filename):
            logger.error(f"\nОШИБКА: Не найден необходимый файл '{filename}'.")
            logger.error("Пожалуйста, сначала запустите соответствующий сценарий оптимизации.")
            return False
    return True

class Optimizer:
    @staticmethod
    def _ensure_dataframe(data, *, nrows=None):
        """Ensure that incoming data is represented as a pandas DataFrame."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, (str, os.PathLike)):
            df = pd.read_csv(data, delimiter=',', nrows=nrows)
            if df.shape[1] == 1:
                df = pd.read_csv(data, delimiter=';', nrows=nrows)
        else:
            raise TypeError(f"Unsupported data type: {type(data)!r}")
        df.columns = df.columns.str.strip()
        return df

    def prepare_structural_data(self, raw_flights, configs_file=CONFIG_FILE):
        """
        Создает структурно полный DataFrame, выбирая ТОЛЬКО ОДНУ
        (самую вместительную) конфигурацию для каждого типа ВС.
        """
        if isinstance(raw_flights, (str, os.PathLike)):
            logger.info(f"  -> Загрузка рейсов из '{raw_flights}' и конфигураций из '{configs_file}'...")
        else:
            logger.info("  -> Работа с dataframe..")
        try:
            flights_df = self._ensure_dataframe(raw_flights)
        except FileNotFoundError:
            logger.error(f"Ошибка: Файл '{raw_flights}' не найден.")
            return None
        except Exception as exc:
            logger.error(f"Кртическая ошибка: {exc}")
            return None

        try:
            configs_df = pd.read_csv(configs_file, delimiter='|')
        except FileNotFoundError:
            logger.error(f"ОШИБКА: Файл '{configs_file}' не найден."); return None
            return None
        except Exception as exc:
            logger.error(f"Кртическая ошибка: {exc}")
            return None

        configs_df.columns = configs_df.columns.str.strip()
        configs_df.rename(columns={'Twn BC': 'Тип ВС'}, inplace=True)
        configs_df.replace('-', '0', inplace=True)
        configs_df[['C', 'W', 'Y']] = configs_df[['C', 'W', 'Y']].astype(int)

        configs_df['total_capacity'] = configs_df['C'] + configs_df['W'] + configs_df['Y']
        configs_df = configs_df.sort_values('total_capacity', ascending=False).drop_duplicates('Тип ВС').sort_index()
        configs_df.drop(columns=['total_capacity'], inplace=True)
        configs_df.replace(0, np.nan, inplace=True)

        merged_df = pd.merge(flights_df, configs_df, on='Тип ВС', how='left')
        id_vars = flights_df.columns.tolist()
        melted_df = pd.melt(merged_df, id_vars=id_vars, value_vars=['C', 'W', 'Y'], var_name='Код кабины', value_name='Емкость кабины')
        final_df = melted_df.dropna(subset=['Емкость кабины'])
        final_df.loc[:, 'Емкость кабины'] = final_df['Емкость кабины'].astype(int)

        final_df.loc[:, 'группа'] = final_df['Номер рейса'].astype(str) + '_' + final_df['Дата вылета'] + '_' + final_df['Время вылета']
        unique_groups = final_df['группа'].unique()
        group_mapping = {group: i + 1 for i, group in enumerate(unique_groups)}
        final_df.loc[:, 'номер_группы'] = final_df['группа'].map(group_mapping)
        final_df.loc[:, '№'] = final_df['номер_группы'].astype(str) + '-' + final_df['Код кабины']

        output_columns = ['№', 'Дата вылета', 'Номер рейса', 'Аэропорт вылета', 'Аэропорт прилета', 'Время вылета', 'Время прилета', 'Емкость кабины', 'Тип ВС', 'Код кабины']
        output_df = final_df[output_columns].sort_values(by='№').reset_index(drop=True)

        logger.info("  -> Структурные данные успешно сгенерированы (без дубликатов).")
        return output_df


    def enrich_from_historical_data(self, df_to_enrich, historical_data_file):
        """
        Дополняет DataFrame данными из эталонного файла по ключу совпадения.
        """
        logger.info(f"  -> Загрузка эталонных данных из '{historical_data_file}'...")
        try:
            df_hist = pd.read_csv(historical_data_file, delimiter=';')
            df_hist.columns = df_hist.columns.str.strip()
        except FileNotFoundError:
            logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: Эталонный файл '{historical_data_file}' не найден.")
            return None

        # Ключи для поиска совпадений
        merge_keys = ['Номер рейса', 'Аэропорт вылета', 'Аэропорт прилета', 'Тип ВС', 'Код кабины']
        # Колонки, которые мы хотим "достроить"
        cols_to_add = ['LF Кабина', 'Бронирования по кабинам', 'Доход пасс', 'Пассажиры']

        # Оставляем в исторической таблице только нужные колонки и первое вхождение по ключу
        df_hist_unique = df_hist[merge_keys + cols_to_add].drop_duplicates(subset=merge_keys, keep='first')

        # Объединяем нашу новую таблицу с историческими данными
        enriched_df = pd.merge(df_to_enrich, df_hist_unique, on=merge_keys, how='left')

        # Проверяем, сколько строк удалось дополнить
        enriched_rows = enriched_df['LF Кабина'].notna().sum()
        total_rows = len(enriched_df)
        logger.info(f"  -> Данные успешно объединены. Дополнено {enriched_rows} из {total_rows} строк.")

        return enriched_df

    def universal_data_preparator(self, raw_data, final_prepared_file=None):
        """
        (НОВАЯ ЛОГИКА) Полный цикл подготовки данных путем поиска в эталонном файле.

        Example:
            >>> optimizer = Optimizer()
            >>> raw_df = pd.read_csv('raw_schedule.csv', delimiter=';')
            >>> prepared_df = optimizer.universal_data_preparator(raw_df)
        """
        logger.info("\n--- Проверка и подготовка исходных данных ---")

        try:
            raw_df = self._ensure_dataframe(raw_data)
        except FileNotFoundError:
            logger.error(f"Не удалось прочитать файл'{raw_data}'.")
            return None
        except Exception as exc:
            logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: Ошибка: {exc}"); return None
            raise

        structural_cols = {'№', 'Код кабины', 'Емкость кабины'}
        prediction_cols = {'LF Кабина', 'Бронирования по кабинам', 'Доход пасс', 'Пассажиры'}

        processed_df = raw_df.copy()

        if not structural_cols.issubset(processed_df.columns):
            logger.warning("-> Требуется подготовка данных. Запуск полного цикла...")
            processed_df = self.prepare_structural_data(raw_df, CONFIG_FILE)
            if processed_df is None:
                return None

        if not prediction_cols.issubset(processed_df.columns):
            logger.warning("-> Обнаружено отсутствие данных о загрузке. Запуск дополнения из эталонного файла...")
            processed_df = self.enrich_from_historical_data(processed_df, HISTORICAL_DATA_FILE)
            if processed_df is None:
                return None

        logger.info(f"-> Финальная обработка: удаление несуществующих классов и форматирование...")
        processed_df = processed_df.copy()
        processed_df['Емкость кабины'] = pd.to_numeric(processed_df['Емкость кабины'], errors='coerce')
        processed_df.dropna(subset=['Емкость кабины'], inplace=True)
        processed_df = processed_df[processed_df['Емкость кабины'] > 0].reset_index(drop=True)

        final_output_columns = ['№', 'Дата вылета', 'Номер рейса', 'Аэропорт вылета', 'Аэропорт прилета', 'Время вылета', 'Время прилета', 'Емкость кабины', 'LF Кабина', 'Бронирования по кабинам', 'Тип ВС', 'Код кабины', 'Доход пасс', 'Пассажиры']

        for col in final_output_columns:
            if col not in processed_df.columns:
                processed_df[col] = np.nan

        processed_df = processed_df[final_output_columns].copy()

        for col in prediction_cols:
            if processed_df[col].dtype == 'object':
                processed_df[col].fillna('0,0', inplace=True)
            else:
                processed_df[col].fillna(0, inplace=True)

        if final_prepared_file:
            logger.info(f"-> Сохранение полностью подготовленных данных в файл '{final_prepared_file}'")
            processed_df.to_csv(final_prepared_file, index=False, sep=';', encoding='utf-8-sig')

        logger.info("--- Подготовка данных завершена ---\n")
        return processed_df

    def overbooking_optimization(self, input_data, output_file=None) -> pd.DataFrame:
        """
        Оптимизация овербукинга с использованием pandas.DataFrame

        Args:
            input_data: pandas.DataFrame или путь к файлу
            output_file: путь для сохранения результата (опционально)

        Returns:
            pandas.DataFrame с оптимизированными данными
        """
        # Преобразуем входные данные в DataFrame если нужно
        if isinstance(input_data, str):
            input_df = pd.read_csv(input_data, delimiter=';', encoding='utf-8-sig')
        else:
            input_df = input_data.copy()

        # Вызываем основную функцию оптимизации
        result_df = run_overbooking_optimization(input_df)

        # Сохраняем результат если указан файл
        if output_file:
            result_df.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')

        return result_df

    def ranking_optimization(self, input_data, output_file=None) -> pd.DataFrame:
        """
        Оптимизация ранжирования с использованием pandas.DataFrame

        Args:
            input_data: pandas.DataFrame или путь к файлу
            output_file: путь для сохранения результата (опционально)

        Returns:
            pandas.DataFrame с оптимизированными данными
        """
        # Преобразуем входные данные в DataFrame если нужно
        if isinstance(input_data, str):
            input_df = pd.read_csv(input_data, delimiter=';', encoding='utf-8-sig')
        else:
            input_df = input_data.copy()

        # Вызываем основную функцию оптимизации
        result_df = run_ranking_optimization(input_df)

        # Сохраняем результат если указан файл
        if output_file:
            result_df.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')

        return result_df

    def form_report(self, raw_file, optimized_file, output_filename, opt_type):
        # Логика формирования отчета

        files_needed = [raw_file, optimized_file]
        if check_files_exist(files_needed):
            file_info = [(files_needed[0], 'Исходный'), (files_needed[1], f'После_оптимизации_{opt_type}')]
            create_comparison_report(file_info, output_filename)
