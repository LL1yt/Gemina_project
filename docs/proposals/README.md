# Предложения Architecture Draft v0.1

Статус: **DRAFT / proposed**, без разрешения исполнения. Эти предложения уточняют механизмы миссии; исходный DOCX сохранён, миссия не заменена.

| Предложение | Документ |
|---|---|
| H=128, embedding=32; p₀=(0.5,0.5,0.5); начальное чтение и одно новое чтение на токен | [Математическая спецификация](../MATHEMATICAL-SPECIFICATION.md) |
| Шаг 0.5 по каждой оси, периодическая трилинейная интерполяция, модуляция U diag(r) V | [Архитектура](../ARCHITECTURE.md) |
| LR 10⁻³/10⁻⁴ и равные LR как контроль, Adam, clipping, короткие синтетические последовательности | [Training design](../TRAINING-DESIGN.md) |
| Будущие этапы sampler → цикл → обучение → inference/replay → контроли | [Outline](../MINIMAL-PROTOTYPE-OUTLINE.md) |
| Несколько агентов, слоёв, режимы записи и согласования | [Research notes](../RESEARCH-NOTES.md) |
| Короткий итеративный цикл, full BPTT/TBPTT, задержки D, разделение W/state/history — исследовательское предложение, NOT_RUN | [Iterative cycle research plan](2026-11-02-SHORT-ITERATIVE-CYCLE-RESEARCH-PLAN.md) |

При принятии отдельного предложения зафиксировать основание и обновить [реестр решений](../decisions/README.md). При замене указать прежнее положение. [Открытые вопросы](../OPEN-QUESTIONS.md) сохраняются до решения.
