# Adaptive Action Token Refinement — OAT Simulation

Исследование посвящено адаптивной глубине декодирования action tokens в OAT.
Практическая часть выполнена как воспроизводимая численная симуляция без GPU и без
внешних Python-зависимостей.

## Что находится в проекте

```text
├── INSTRUCTIONS.md                ← пошаговое воспроизведение эксперимента (читать первым)
├── README.md                      ← этот файл
├── report_final.tex               ← аналитический отчёт (LaTeX)
├── simulation/
│   ├── simulate_adaptive_oat.ps1  ← основной запуск (PowerShell)
│   ├── simulate_adaptive_oat.py   ← основной запуск (Python 3.8+)
│   ├── ablation_eps.ps1           ← ablation study (PowerShell)
│   ├── ablation_eps.py            ← ablation study (Python)
│   └── README.md                  ← описание модели и параметров
├── results/
│   ├── summary.json               ← агрегированная статистика
│   ├── episodes.csv               ← поэпизодные данные (300 эпизодов)
│   └── ablation_eps.json          ← ablation по eps ∈ {0.02, 0.05, 0.10, 0.15, 0.20}
└── logs/
    └── assistant_interaction_logs.md  ← логи работы с ассистентами
```

## Быстрый старт

Из папки `Git_project`:

```powershell
powershell -ExecutionPolicy Bypass -File .\simulation\simulate_adaptive_oat.ps1
powershell -ExecutionPolicy Bypass -File .\simulation\ablation_eps.ps1
```

После запуска должны появиться или обновиться файлы:

- `results/summary.json`
- `results/episodes.csv`
- `results/ablation_eps.json`

Подробный порядок воспроизведения приведен в `INSTRUCTIONS.md`.

## Референсный запуск из отчета

Параметры референсного эксперимента:

- `seed = 42`
- `NEpisodes = 100` на каждый тип задачи
- `NSteps = 20`
- `eps = 0.05`
- всего `300` эпизодов

Ключевые результаты из `results/summary.json`:

| Задача | Средние токены | Экономия vs OAT8 | Adaptive error |
|---|---|---|---|
| simple | 3.03 / 8 | 62.1% | 0.01792 |
| complex | 6.66 / 8 | 16.7% | 0.07851 |
| mixed | 5.64 / 8 | 29.5% | 0.06113 |
| all | 5.11 / 8 | 36.1% | 0.05252 |

## Зачем два режима запуска

- `PowerShell` режим нужен для воспроизведения на Windows без установки Python-пакетов.
- `Python` режим удобен для чтения и модификации кода симуляции.

Оба режима генерируют один и тот же набор артефактов в `results/`.

## Ссылки

- [BLT](https://arxiv.org/abs/2412.09871)
- [H-Net](https://arxiv.org/abs/2507.07955)
- [OAT](https://arxiv.org/abs/2602.04215)
- [OAT GitHub](https://github.com/Chaoqi-LIU/oat)
