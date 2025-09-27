# API v1

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
| `X-Session-Id` | string | ✅ Да | Идентификатор сессии пользователя, полученный из [`/api/session/me`](#get-apisessionme). |

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
    ...
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
    ...
  ],
  "iframes": [
    {
      "id": "frame_reports",
      "title": "Отчёт по маршрутам",
      "src": "https://example.com/report"
    },
    ...
  ]
}
```

**Response 404**
```json
{ "detail": "Session not found" }
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

**Query params**
| Параметр      | Тип    | Обязательно | Описание |
|---------------|--------|-------------|----------|
| `stored_name` | string | Да  | Уникальное имя файла (из upload-ответа) |
| `download_as` | string | Нет | Имя файла при скачивании |

**Response 200**
- Бинарный поток файла (`application/octet-stream`).

**Response 404**
```json
{
  "error": "Файл не найден"
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

## f



