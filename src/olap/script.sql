-- Создание справочных таблиц (те, на которые ссылаются)

-- 1. Таблица для номеров рейсов
-- Соответствует схеме: "№ рейса"
CREATE TABLE flight_numbers (
    id SERIAL PRIMARY KEY,                      -- Поле: Код
    flight_no VARCHAR(10) NOT NULL UNIQUE       -- Поле: Номер рейса
);

-- 2. Таблица для аэропортов
-- Соответствует схеме: "Аэропорт"
CREATE TABLE airports (
    id SERIAL PRIMARY KEY,                      -- Поле: Код
    -- Предполагается, что "Аэропорт" - это код IATA (напр., 'SVO')
    airport_code VARCHAR(3) NOT NULL UNIQUE     -- Поле: Аэропорт
);

-- 3. Таблица для воздушных судов (самолетов)
-- Соответствует схеме: "ВС" (Воздушное Судно)
CREATE TABLE aircrafts (
    id SERIAL PRIMARY KEY,                      -- Поле: Код
    aircraft_type VARCHAR(50) NOT NULL,         -- Поле: Тип ВС
    cabin_code VARCHAR(10) NOT NULL,            -- Поле: Код кабины
    cabin_capacity INTEGER NOT NULL CHECK (cabin_capacity > 0) -- Поле: Емкость кабины
);


-- Создание основных таблиц с данными о загрузке

-- 4. Таблица "Загрузка Рейсов"
-- Соответствует схеме: "ЗагрузкаРейсов"
CREATE TABLE flight_loads (
    id SERIAL PRIMARY KEY,                                      -- Поле: Код
    departure_date DATE NOT NULL,                               -- Поле: ДатаВылета
    departure_time TIME NOT NULL,                               -- Поле: ВремяВылета
    arrival_time TIME NOT NULL,                                 -- Поле: ВремяПрилета
    cabin_load_factor NUMERIC(5, 2),                            -- Поле: LFКабина (Load Factor в %)
    bookings INTEGER NOT NULL,                                  -- Поле: Бронирования
    revenue NUMERIC(12, 2) NOT NULL,                            -- Поле: Доход
    passengers INTEGER NOT NULL,                                -- Поле: Пассажиры

    -- Внешние ключи для связи с другими таблицами
    flight_number_id INTEGER NOT NULL REFERENCES flight_numbers(id),    -- Связь с "№ рейса" через НомерРейса
    departure_airport_id INTEGER NOT NULL REFERENCES airports(id),      -- Связь с "Аэропорт" через АэропортВылета
    arrival_airport_id INTEGER NOT NULL REFERENCES airports(id),        -- Связь с "Аэропорт" через АэропортПрилета
    aircraft_id INTEGER NOT NULL REFERENCES aircrafts(id),              -- Связь с "ВС" через ТипВС

    -- Дополнительное ограничение: аэропорт вылета не может быть равен аэропорту прилета
    CONSTRAINT chk_different_airports CHECK (departure_airport_id <> arrival_airport_id)
);

-- 5. Таблица "Загрузка Рейсов Обновленная"
-- Соответствует схеме: "ЗагрузкаРейсовОбно..."
CREATE TABLE flight_loads_updated (
    id SERIAL PRIMARY KEY,                                      -- Поле: Код
    departure_date DATE NOT NULL,                               -- Поле: ДатаВылета
    departure_time TIME NOT NULL,                               -- Поле: ВремяВылета
    arrival_time TIME NOT NULL,                                 -- Поле: ВремяПрилета
    cabin_load_factor NUMERIC(5, 2),                            -- Поле: LFКабина
    bookings INTEGER NOT NULL,                                  -- Поле: Бронирования
    revenue NUMERIC(12, 2) NOT NULL,                            -- Поле: Доход
    passengers INTEGER NOT NULL,                                -- Поле: Пассажиры

    -- Внешние ключи
    flight_number_id INTEGER NOT NULL REFERENCES flight_numbers(id),
    departure_airport_id INTEGER NOT NULL REFERENCES airports(id),
    arrival_airport_id INTEGER NOT NULL REFERENCES airports(id),
    aircraft_id INTEGER NOT NULL REFERENCES aircrafts(id),

    -- Дополнительное ограничение
    CONSTRAINT chk_different_airports_updated CHECK (departure_airport_id <> arrival_airport_id)
);

-- Для удобства можно создать индексы для внешних ключей,
-- это ускорит операции соединения (JOIN) таблиц.
CREATE INDEX idx_flight_loads_flight_number ON flight_loads(flight_number_id);
CREATE INDEX idx_flight_loads_departure_airport ON flight_loads(departure_airport_id);
CREATE INDEX idx_flight_loads_arrival_airport ON flight_loads(arrival_airport_id);
CREATE INDEX idx_flight_loads_aircraft ON flight_loads(aircraft_id);

CREATE INDEX idx_flight_loads_updated_flight_number ON flight_loads_updated(flight_number_id);
CREATE INDEX idx_flight_loads_updated_departure_airport ON flight_loads_updated(departure_airport_id);
CREATE INDEX idx_flight_loads_updated_arrival_airport ON flight_loads_updated(arrival_airport_id);
CREATE INDEX idx_flight_loads_updated_aircraft ON flight_loads_updated(aircraft_id);