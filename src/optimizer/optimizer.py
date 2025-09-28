import logging
import os
from logging.config import dictConfig
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

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
            print(f"\nОШИБКА: Не найден необходимый файл '{filename}'.")
            print("Пожалуйста, сначала запустите соответствующий сценарий оптимизации.")
            return False
    return True

class Optimizer:
    def overbooking_optimization(self, input_file, output_file) -> bool:
        run_overbooking_optimization(initial_schedule_file=input_file, final_output_file=output_file)

    def ranking_optimization(self, input_file, output_file) -> bool:
        run_ranking_optimization(data_file=input_file, optimized_file=output_file)

    def form_report(self, raw_file, optimized_file, output_filename, opt_type):
        # Логика формирования отчета

        files_needed = [raw_file, optimized_file]
        if check_files_exist(files_needed):
            file_info = [(files_needed[0], 'Исходный'), (files_needed[1], f'После_оптимизации_{opt_type}')]
            create_comparison_report(file_info, output_filename)
