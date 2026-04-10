# Simulation Scripts

## Требования

- Python 3.8+ и только стандартная библиотека
- либо Windows PowerShell без дополнительных зависимостей

## Референсный запуск

Именно эти параметры использованы в отчете:

- `seed = 42`
- `100` эпизодов на каждый тип задач
- `20` шагов на эпизод
- `eps = 0.05`

### Python

```bash
python simulate_adaptive_oat.py --seed 42 --n_episodes 300 --n_steps 20 --eps 0.05
python ablation_eps.py --seed 42 --n_episodes 300 --n_steps 20
```

### PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File .\simulate_adaptive_oat.ps1 `
    -Seed 42 -NEpisodes 100 -NSteps 20 -Eps 0.05

powershell -ExecutionPolicy Bypass -File .\ablation_eps.ps1 `
    -Seed 42 -NEpisodes 100 -NSteps 20
```

## Описание модели

`simulate_adaptive_oat.py` и `simulate_adaptive_oat.ps1` реализуют одну и ту же
синтетическую модель:

1. **Траектории**:
   - `simple`: малая амплитуда движений, complexity ∈ [0.05, 0.25]
   - `complex`: большая амплитуда движений, complexity ∈ [0.60, 0.95]
   - `mixed`: переменная активность по синусоидальному профилю

2. **VQ-кодирование**:
   - ошибка реконструкции убывает геометрически при добавлении токенов
   - для простых действий decay быстрее
   - для сложных действий decay медленнее

3. **Критерий остановки**:

   ```text
   stop when err_k < eps * ||action||
   ```

   Это и есть эвристический adaptive stopping для OAT.

4. **Метрики**:
   - `OAT1`, `OAT2`, `OAT4`, `OAT8` reconstruction error
   - `adaptive_error`
   - `adaptive_tokens_mean`
   - `token_saving_pct`
   - `quality_delta_vs_oat8_pct`

## Параметры

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--seed` | 42 | seed для воспроизводимости |
| `--n_episodes` | 300 | общее число эпизодов в Python-версии |
| `--n_steps` | 20 | шагов в эпизоде |
| `--eps` | 0.05 | порог остановки |
| `--out_dir` | `../results` | директория для результатов |

PowerShell-версия использует `NEpisodes = 100` как число эпизодов на каждый тип задач.

## Результаты

Скрипты генерируют артефакты в `../results/`:

- `episodes.csv` — поэпизодные данные
- `summary.json` — агрегированная статистика основного запуска
- `ablation_eps.json` — агрегированная статистика по разным `eps`

## Ожидаемые значения для проверки

Для основного запуска с `eps=0.05`:

- `all.adaptive_tokens_mean = 5.11`
- `all.token_saving_pct = 36.1`
- `simple.adaptive_tokens_mean = 3.03`
- `complex.adaptive_tokens_mean = 6.66`

Для ablation лучшее компромиссное значение в текущем эксперименте — `eps = 0.05`.
