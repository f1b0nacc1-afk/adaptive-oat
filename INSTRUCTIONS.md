# Инструкция по запуску

Ниже приведен точный порядок, по которому можно повторить эксперимент и получить
те же типы артефактов, что используются в отчете.

## 1. Что нужно для воспроизведения

Минимальный вариант:

- Windows PowerShell
- либо Python 3.8+ без дополнительных пакетов

Для симуляции не нужны `pip install`, GPU, MuJoCo или LIBERO.

## 2. Откуда запускать

Откройте PowerShell в папке `Git_project`.

Все команды ниже предполагают, что текущая директория:

```powershell
cd C:\Users\plane\Documents\ML\CKM\Git_project
```

## 3. Основной эксперимент

Это тот запуск, на который ссылается отчет.

Параметры:

- `Seed = 42`
- `NEpisodes = 100` на каждый тип задач
- `NSteps = 20`
- `Eps = 0.05`

### Вариант A: PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File .\simulation\simulate_adaptive_oat.ps1 `
    -Seed 42 -NEpisodes 100 -NSteps 20 -Eps 0.05 -OutDir .\results
```

### Вариант B: Python

```powershell
python .\simulation\simulate_adaptive_oat.py `
    --seed 42 --n_episodes 300 --n_steps 20 --eps 0.05 --out_dir .\results
```

Примечание: в PowerShell-параметре `NEpisodes=100` означает 100 эпизодов на каждый
тип задач, то есть всего 300 эпизодов. В Python-версии общее число эпизодов
задается напрямую через `--n_episodes 300`.

## 4. Ablation study по eps

Этот запуск формирует таблицу выбора порога остановки.

### Вариант A: PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File .\simulation\ablation_eps.ps1 `
    -Seed 42 -NEpisodes 100 -NSteps 20 -OutDir .\results
```

### Вариант B: Python

```powershell
python .\simulation\ablation_eps.py `
    --seed 42 --n_episodes 300 --n_steps 20 --out_dir .\results
```

Python-скрипт прогоняет `eps ∈ {0.02, 0.05, 0.10, 0.15, 0.20}`.

## 5. Какие файлы должны получиться

После основного запуска:

- `results/episodes.csv`
- `results/summary.json`

После ablation:

- `results/ablation_eps.json`

## 6. Как проверить, что все воспроизвелось корректно

Откройте `results/summary.json` и проверьте ключевые значения:

- `all.adaptive_tokens_mean = 5.11`
- `all.token_saving_pct = 36.1`
- `all.adaptive_error = 0.05252`
- `simple.adaptive_tokens_mean = 3.03`
- `complex.adaptive_tokens_mean = 6.66`

Откройте `results/ablation_eps.json` и проверьте строку для `0.05`:

- `all.adaptive_tokens_mean = 5.11`
- `all.token_saving_pct = 36.1`
- `all.adaptive_error = 0.052519`

Небольшие отличия в последних знаках после запятой допустимы только при
самостоятельной модификации кода. Для исходного кода этого проекта результаты
должны совпадать.

## 7. Параметры симуляции

| Параметр | По умолчанию | Описание |
|---|---|---|
| `Seed` / `--seed` | 42 | seed для воспроизводимости |
| `NEpisodes` | 100 | эпизодов на каждый тип задач в PowerShell-версии |
| `--n_episodes` | 300 | общее число эпизодов в Python-версии |
| `NSteps` / `--n_steps` | 20 | шагов в одном эпизоде |
| `Eps` / `--eps` | 0.05 | порог остановки: `stop when err < eps * action_norm` |
| `OutDir` / `--out_dir` | `results` | директория для артефактов |

Рекомендуемая интерпретация `eps`:

- `0.02` — консервативный режим, близко к OAT8 по качеству
- `0.05` — лучший баланс качества и экономии
- `0.10+` — агрессивная экономия с заметной потерей качества

## 8. Про папку `oat/` и почему её нет в этом репозитории

`oat/` — это клон оригинального репозитория авторов метода:
**[github.com/Chaoqi-LIU/oat](https://github.com/Chaoqi-LIU/oat)**

В нём нет ни одного нашего изменения. Мы его изучили, установили окружение
(`.venv` весит ~5.5 GB) и разобрались в архитектуре — всё это отражено в логах
и отчёте. Класть его в свой репо неправильно: это чужой код, а `.venv` никакой
Git-хостинг не примет.

Наш вклад — численная симуляция из `simulation/`, которая верифицирует
центральную гипотезу без GPU и без внешних зависимостей.

---

## 9. Полный OAT baseline (требует GPU + Linux/WSL)

Если хочется запустить настоящий OAT на LIBERO, вот пошаговые команды
прямо из официального README:

### Шаг 1 — клонировать с подмодулями

```bash
git clone --recurse-submodules git@github.com:Chaoqi-LIU/oat.git
cd oat
```

### Шаг 2 — установить окружение

```bash
# Вариант A: uv (рекомендован авторами)
uv sync
uv pip install -e .

# Вариант B: micromamba
micromamba env create -f conda_env.yaml
```

### Шаг 3 — скачать датасет LIBERO

```bash
# Готовый датасет с HuggingFace (быстрее):
# https://huggingface.co/datasets/chaoqi-liu/libero10_N500.zarr

# Или собрать локально:
uv run third_party/LIBERO/benchmark_scripts/download_libero_datasets.py \
    --datasets libero_spatial

uv run scripts/convert_libero_dataset.py \
    --root_dir data/libero --hdf5_dir_name hdf5_datasets

uv run scripts/compose_libero_multitask_dataset.py \
    --multitask_name libero10 --root_dir data/libero
```

### Шаг 4 — обучить tokenizer

```bash
HYDRA_FULL_ERROR=1 uv run accelerate launch \
    --num_machines 1 --multi_gpu --num_processes 1 \
    scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=libero/libero10
```

### Шаг 5 — обучить policy

```bash
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run accelerate launch \
    --num_machines 1 --multi_gpu --num_processes 1 \
    scripts/run_workspace.py \
    --config-name=train_oatpolicy \
    task/policy=libero/libero10 \
    task.policy.lazy_eval=false \
    policy.action_tokenizer.checkpoint=outputs/train_oattok/oattok.ckpt
```

### Шаг 6 — оценка

```bash
uv run scripts/eval_policy_sim.py \
    --checkpoint outputs/train_oatpolicy/oatpolicy.ckpt \
    --output_dir output/eval/libero10 \
    --num_exp 5
```

### Куда добавить adaptive stopping

После получения baseline-результатов можно добавить критерий остановки
из нашей гипотезы. Точка вмешательства — `oat/oat/policy/oat_policy.py`,
метод `predict_action()`, авторегрессивный цикл декодирования:

```python
# Псевдокод модификации inference loop
action_norm = action_embedding.norm()
threshold = eps * action_norm   # eps = 0.05 по результатам ablation

tokens = []
for k in range(1, max_tokens + 1):
    tokens.append(generate_next_token(tokens))
    decoded = tokenizer.decode_prefix(tokens)
    err = reconstruction_error(decoded, action_norm)
    if err < threshold:
        break   # ← ранняя остановка

return decoded
```

Гипотеза: этот change не требует переобучения — только изменения inference.
