# API v1

## 🔐 Auth

### POST `/api/auth/login`
Авторизация пользователя и получение JWT токена. Единственный валидный логин — `admin`. Пароль берётся из переменной окружения `.env` → `ADMIN_PASSWORD`.

**Form data (x-www-form-urlencoded)**
| Поле       | Тип   | Обязательно | Описание |
|------------|-------|--------------|----------|
| `username` | string | Да | Имя пользователя. Для данной версии — всегда `admin`. |
| `password` | string | Да | Пароль администратора (`ADMIN_PASSWORD` из `.env`). |

**Response 200**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

**Response 401**
```json
{
  "detail": "Incorrect username or password"
}
```

---

### GET `/api/auth/me`
Получить профиль текущего пользователя по Bearer-токену.

**Headers**
| Заголовок          | Тип    | Обязательно | Описание |
|--------------------|--------|--------------|----------|
| `Authorization`    | string | Да | `Bearer <JWT>` полученный из [`/api/auth/login`](#post-apiauthlogin). |

**Response 200**
```json
{
  "id": 1,
  "username": "admin",
  "full_name": null,
  "role": "admin",
  "created_at": "2025-09-27T12:00:00Z"
}
```

**Response 401**
```json
{
  "detail": "Could not validate credentials"
}
```

## Session

### GET `/api/v1/session/me`
Получить текущий идентификатор сессии (`X-Session-Id`).
Если заголовок отсутствует — сервер создаёт новый `X-Session-Id` и возвращает его в ответе.

**Headers**
| Заголовок | Тип | Обязательно | Описание |
|------------|-----|--------------|-----------|
| `X-Session-Id` | string | Нет | Уникальный идентификатор сессии пользователя. Если не передан — сервер создаёт новый. |

**Response 200**
```json
{
  "session_id": "sess_5zQ0nVrN8e8SxA1TqFlJxQ"
}
```

**Response Headers**
| Заголовок | Описание |
|------------|-----------|
| `X-Session-Id` | Текущий идентификатор сессии (новый или существующий) |

### GET `/api/v1/session/data`
Получить **все данные** текущей сессии по идентификатору `X-Session-Id`.

Возвращает **полный объект сессии**: основные метрики, оба расписания **без обрезки** (все элементы массива), а также список `iframes` (каждый с `id`).

**Headers**
| Заголовок | Тип | Обязательно | Описание |
|---|---|---|---|
| `X-Session-Id` | string | Да | Идентификатор сессии пользователя, полученный из [`/api/session/me`](#get-apisessionme). |

**Response 200**
```json
{
  "session_id": "sess_5zQ0nVrN8e8SxA1TqFlJxQ",
  "expires_at": "2025-09-28T12:00:00Z",
  "main_metrics": {
    "income": { "value": 12000, "optimized_value": 13500 },
    "passengers": { "value": 240, "optimized_value": 270 },
    "avg_check": { "value": 50, "optimized_value": 55 }
  },
  "unoptimized_schedule": [
    {
      "date": "2025-04-18",
      "flight_number": "224",
      "dep_airport": "SVO",
      "arr_airport": "SYX",
      "dep_time": "10:35",
      "arr_time": "01:15",
      "flight_capacity": 28,
      "lf_cabin": 0.6786,
      "cabins_brones": 19,
      "flight_type": "359",
      "cabin_code": "C",
      "pass_income": 10048.02,
      "passengers": 19
    },
    //...
  ],
  "optimized_schedule": [
    {
      "date": "2025-04-18",
      "flight_number": "224",
      "dep_airport": "SVO",
      "arr_airport": "SYX",
      "dep_time": "10:35",
      "arr_time": "01:15",
      "flight_capacity": 24,
      "lf_cabin": 0.25,
      "cabins_brones": 6,
      "flight_type": "359",
      "cabin_code": "W",
      "pass_income": 2013.02,
      "passengers": 6
    },
    //...
  ],
  "iframes": [
    {
      "id": "frame_reports",
      "title": "Отчёт по маршрутам",
      "src": "https://example.com/report"
    },
    //...
  ]
}
```

**Response 404**
```json
{ "detail": "Session not found" }
```

### GET `/api/v1/session/metrics?id={metric_type}`

Получить конкретную метрику сессии (`income`, `passengers`, `avg_check`) по её типу.
Метрика возвращается в формате пары значений: текущее (`value`) и оптимизированное (`optimized_value`).

**Headers**
| Заголовок | Тип | Обязательно | Описание |
|------------|-----|--------------|-----------|
| `X-Session-Id` | string | Да | Уникальный идентификатор активной сессии, полученный из [`/api/session/me`](#get-apisessionme). |

**Query Parameters**
| Параметр | Тип | Обязательно | Описание |
|-----------|-----|--------------|-----------|
| `id` | string | Да | Тип метрики: `income`, `passengers` или `avg_check`. |

**Response 200**
```json
{
  "id": "income",
  "value": 12000,
  "optimized_value": 13500
}
```

**Response 400**
```json
{
  "detail": "Invalid metric type. Must be one of: income, passengers, avg_check."
}
```

**Response 404**
```json
{
  "detail": "Session or metric not found."
}
```

### GET `/api/v1/session/iframe?id={iframe_id}`

Получить данные конкретного `iframe` по его идентификатору (`id`) из текущей сессии.

**Headers**
| Заголовок | Тип | Обязательно | Описание |
|------------|-----|--------------|-----------|
| `X-Session-Id` | string | Да | Уникальный идентификатор активной сессии, полученный из [`/api/session/me`](#get-apisessionme). |

**Query Parameters**
| Параметр | Тип | Обязательно | Описание |
|-----------|-----|--------------|-----------|
| `id` | string | Да | Идентификатор iframe (например, `frame_reports`, `frame-map`). |

**Response 200**
```json
{
  "id": "frame_reports",
  "title": "Отчёт по маршрутам",
  "src": "https://example.com/report"
}
```

**Response 404**
```json
{
  "detail": "Iframe not found."
}
```

## Files

### POST `/api/v1/files/upload`
Загрузить файл на сервер.

**Headers**
| Заголовок | Тип | Обязательно | Описание |
|------------|-----|--------------|-----------|
| `X-Session-Id` | string | Нет | Уникальный идентификатор сессии пользователя. Если не передан — сервер создаёт новый. |

**Query params**
| Параметр     | Тип    | Обязательно | Описание |
|--------------|--------|-------------|----------|
| `as_name`    | string | Нет | Имя, под которым сохранить файл |

**Form-data**
| Параметр | Тип  | Обязательно | Описание |
|----------|------|-------------|----------|
| `file`   | file | Да | Загружаемый бинарный файл |

**Response 200**
```json
{
  "session_id": "sess_5zQ0nVrN8e8SxA1TqFlJxQ",
  "file_name": "report.csv",
  "stored_name": "8f3a..._report.csv",
  "size_bytes": 1048576,
  "download_url": "/api/files/download?session_id=sess_5zQ0nVrN8e8SxA1TqFlJxQ&stored_name=8f3a..._report.csv"
}
```

---

### GET `/api/v1/files/download`
Скачать файл по `stored_name`.

**Headers**
| Заголовок | Тип | Обязательно | Описание |
|------------|-----|--------------|-----------|
| `X-Session-Id` | string | Нет | Уникальный идентификатор сессии пользователя. Если не передан — сервер создаёт новый. |

**Response 200**
- Бинарный поток файла (`application/octet-stream`).

**Response 404**
```json
{
  "error": "No file_key in session"
}
```

**Response 410**
```json
{
  "error": "File no longer available"
}
```

---

### DELETE `/api/v1/files/delete`
Удалить файл по `stored_name`.

**Headers**
| Заголовок | Тип | Обязательно | Описание |
|------------|-----|--------------|-----------|
| `X-Session-Id` | string | Нет | Уникальный идентификатор сессии пользователя. Если не передан — сервер создаёт новый. |

**Query params**
| Параметр      | Тип    | Обязательно | Описание |
|---------------|--------|-------------|----------|
| `stored_name` | string | Да | Уникальное имя файла (из upload-ответа) |

**Response 200**
```json
{
  "session_id": "sess_5zQ0nVrN8e8SxA1TqFlJxQ",
  "stored_name": "8f3a..._report.csv",
  "deleted": true,
  "size_bytes": 1048576
}
```

**Response 404**
```json
{
  "error": "Файл не найден"
}
```

## Data Management

### POST `/api/v1/data/import-csv`
Загрузить и прочитать данные из CSV файла.

**Headers**
| Заголовок | Тип | Обязательно | Описание |
|------------|-----|--------------|-----------|
| `X-Session-Id` | string | Нет | Уникальный идентификатор сессии пользователя. Если не передан — сервер создаёт новый. |

**Response 200**
```json
{
  "session_id": "sess_f9e2d8c1a2b04f5bb7b123",
  "rows_parsed": 15420,
  "columns": ["date", "flight_number", "origin", "destination", "cabin_code", "passengers", "revenue"]
}
```

---

### POST `/api/v1/data/build-olap`
Сформировать OLAP-структуру (витрины) из загруженных данных.

**Headers**
| Заголовок | Тип | Обязательно | Описание |
|------------|-----|--------------|-----------|
| `X-Session-Id` | string | Да | Идентификатор сессии пользователя |

**Response 200**
```json
{
  "session_id": "sess_f9e2d8c1a2b04f5bb7b123",
  "olap_tables": [
    "flights_by_date",
    "flights_by_airport",
    "flights_by_cabin"
  ],
  "status": "created"
}
```

---

### POST `/api/v1/data/upload-datalens`
Загрузить OLAP-таблицы в Yandex DataLens для построения графиков.

**Headers**
| Заголовок | Тип | Обязательно | Описание |
|------------|-----|--------------|-----------|
| `X-Session-Id` | string | Да | Идентификатор сессии пользователя |

**Query params**
| Параметр   | Тип    | Обязательно | Описание |
|------------|--------|-------------|-----------|
| `dataset`  | string | Да | Имя набора данных для загрузки в DataLens |

**Response 200**
```json
{
  "session_id": "sess_f9e2d8c1a2b04f5bb7b123",
  "dataset": "optimized_flights",
  "status": "uploaded",
  "datalens_url": "https://datalens.yandex.ru/dashboard/abcd1234"
}
```
