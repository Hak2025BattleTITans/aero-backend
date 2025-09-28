# main.py - файл для вызова алгоритмов и составления эксель отчетов
import time
import os
import main_ranking
import overbooking_optimizer
import report_generator

def display_menu():
    """Отображает меню выбора."""
    print("\n" + "="*60)
    print("      ИНТЕРАКТИВНЫЙ ОПТИМИЗАТОР РАСПИСАНИЯ")
    print("="*60)
    print("--- Оптимизация ---")
    print("  0 - Только оптимизация слотов (ranking)")
    print("  1 - Только оптимизация овербукинга")
    print("  2 - Сначала слоты (ranking), затем овербукинга")
    print("  !3 - Сначала овербукинг, затем слоты (ranking)")
    print("\n--- Создание отчетов ---")
    print("  4 - Отчет для сценария 0 (Ranking vs Исходный)")
    print("  5 - Отчет для сценария 1 (Overbooking vs Исходный)")
    print("  6 - Отчет для сценария 2 (Ranking -> Overbooking)")
    print("  !7 - Отчет для сценария 3 (Overbooking -> Ranking)")
    print("="*60)
    print("Нажмите Ctrl+C для выхода.")

def check_files_exist(filenames):
    """Проверяет, существуют ли все файлы в списке."""
    for filename in filenames:
        if not os.path.exists(filename):
            print(f"\nОШИБКА: Не найден необходимый файл '{filename}'.")
            print("Пожалуйста, сначала запустите соответствующий сценарий оптимизации.")
            return False
    return True

def main():
    """Основная функция, управляющая сценариями."""
    INITIAL_DATA_FILE = 'hackathon_data_main_with_numbers.csv'
    if not os.path.exists(INITIAL_DATA_FILE):
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не найден основной файл данных '{INITIAL_DATA_FILE}'. Программа не может продолжить работу.")
        return

    while True:
        display_menu()
        try:
            choice_str = input("Введите номер сценария (0-7): ")
            if not choice_str: continue
            choice = int(choice_str)
        except ValueError:
            print("\nОшибка: Введено не число. Пожалуйста, попробуйте снова.")
            continue
        except KeyboardInterrupt:
            print("\n\nПолучен сигнал выхода. Завершение программы.")
            break

        start_time = time.time()

        # ======
        # Сценарий 3: Овербукинг -> Ранжирование
        # ======

        if choice == 3:
            print("\n--- Сценарий 3: Овербукинг -> Ранжирование ---")
            intermediate_file = 'final_schedule_o3.csv'
            final_file = 'final_schedule_o_r3.csv'

            # Запуск алгоритмов

            # Оптимизация овербукинга
            print("\n[Шаг 1/2] Оптимизация овербукинга...")
            result_step1 = overbooking_optimizer.run_overbooking_optimization(initial_schedule_file=INITIAL_DATA_FILE, final_output_file=intermediate_file)

            # Проверка успешности первого шага
            if not result_step1:
                print("\nОшибка на первом шаге. Прерывание сценария.")
                break

            # Оптимизация слотов
            print("\n[Шаг 2/2] Оптимизация слотов...")
            main_ranking.run_ranking_optimization(data_file=intermediate_file, optimized_file=final_file)

            # Формирование отчёта
            files_needed = [INITIAL_DATA_FILE, 'final_schedule_o3.csv', 'final_schedule_o_r3.csv']
            if check_files_exist(files_needed):
                file_info = [(files_needed[0], 'Исходный'), (files_needed[1], 'После_оптимизации_овербукинга'), (files_needed[2], 'После_оптимизаций_овербукинга_перемещений')]
                report_generator.create_comparison_report(file_info, 'Report_o_r3.xlsx')

        # ======
        # Сецнарий 1 и 2 (ранжирование и овербукинг)
        # ======


        elif choice == 0:
            # Запуск только ранжирования
            print("\n--- Сценарий 0: Только оптимизация слотов ---")
            output_file = 'final_schedule_ranking0.csv'
            main_ranking.run_ranking_optimization(data_file=INITIAL_DATA_FILE, optimized_file=output_file)

            # Формирование отчёта
            files_needed = [INITIAL_DATA_FILE, 'final_schedule_ranking0.csv']
            if check_files_exist(files_needed):
                file_info = [(files_needed[0], 'Исходный'), (files_needed[1], 'После_оптимизации_перемещений')]
                report_generator.create_comparison_report(file_info, 'Report_ranking0.xlsx')

        elif choice == 1:
            # Запуск только овербукинга
            print("\n--- Сценарий 1: Только оптимизация овербукинга ---")
            output_file = 'final_schedule_overbooking1.csv'
            overbooking_optimizer.run_overbooking_optimization(initial_schedule_file=INITIAL_DATA_FILE, final_output_file=output_file)

            # Формирование отчёта
            files_needed = [INITIAL_DATA_FILE, 'final_schedule_overbooking1.csv']
            if check_files_exist(files_needed):
                file_info = [(files_needed[0], 'Исходный'), (files_needed[1], 'После_оптимизации_овербукинга')]
                report_generator.create_comparison_report(file_info, 'Report_overbooking1.xlsx')

        # =====
        # Другие сценарии - отчеты
        # =====

        elif choice == 2:
            print("\n--- Сценарий 2: Ранжирование -> Овербукинг ---")
            intermediate_file = 'final_schedule_r2.csv'; final_file = 'final_schedule_r_o2.csv'
            print("\n[Шаг 1/2] Оптимизация слотов..."); result_step1 = main_ranking.run_ranking_optimization(data_file=INITIAL_DATA_FILE, optimized_file=intermediate_file)
            if result_step1:
                print("\n[Шаг 2/2] Оптимизация овербукинга..."); overbooking_optimizer.run_overbooking_optimization(initial_schedule_file=intermediate_file, final_output_file=final_file)
            else: print("\nОшибка на первом шаге. Прерывание сценария.")



        elif choice == 4:
            files_needed = [INITIAL_DATA_FILE, 'final_schedule_ranking0.csv']
            if check_files_exist(files_needed):
                file_info = [(files_needed[0], 'Исходный'), (files_needed[1], 'После_оптимизации_перемещений')]
                report_generator.create_comparison_report(file_info, 'Report_ranking0.xlsx')

        elif choice == 5:
            files_needed = [INITIAL_DATA_FILE, 'final_schedule_overbooking1.csv']
            if check_files_exist(files_needed):
                file_info = [(files_needed[0], 'Исходный'), (files_needed[1], 'После_оптимизации_овербукинга')]
                report_generator.create_comparison_report(file_info, 'Report_overbooking1.xlsx')

        elif choice == 6:
            files_needed = [INITIAL_DATA_FILE, 'final_schedule_r2.csv', 'final_schedule_r_o2.csv']
            if check_files_exist(files_needed):
                file_info = [(files_needed[0], 'Исходный'), (files_needed[1], 'После_оптимизации_перемещений'), (files_needed[2], 'После_оптимизаций_перемещений_овербукинга')]
                report_generator.create_comparison_report(file_info, 'Report_r_o2.xlsx')

        elif choice == 7:
            files_needed = [INITIAL_DATA_FILE, 'final_schedule_o3.csv', 'final_schedule_o_r3.csv']
            if check_files_exist(files_needed):
                file_info = [(files_needed[0], 'Исходный'), (files_needed[1], 'После_оптимизации_овербукинга'), (files_needed[2], 'После_оптимизаций_овербукинга_перемещений')]
                report_generator.create_comparison_report(file_info, 'Report_o_r3.xlsx')

        else:
            print(f"\nОшибка: Неверный номер сценария '{choice}'. Доступные варианты: 0-7.")

        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("\n" + "="*60)
        print(f"Выполнение сценария {choice} завершено.")
        print(f"Время работы: {minutes} мин {seconds} сек.")
        print("="*60)

if __name__ == "__main__":
    main()