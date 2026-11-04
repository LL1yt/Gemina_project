# Gemina — M0 Implementation Passport v0.1

**Дата решения:** 2026-11-04. **Статус:** OWNER-ALIGNED DESIGN BASELINE / implementation **NOT_RUN**.  
**Цель:** согласовать необходимые выборы из [PR #2](https://github.com/LL1yt/Gemina_project/pull/2) (head `8b6c2f6`) и [PR #5](https://github.com/LL1yt/Gemina_project/pull/5) (head `40811f3`) для единой первой реализации. [MVP Baseline v0.1](MVP-BASELINE-v0.1.md) остаётся границей приоритета. Это не merge исходных веток и не разрешение на сборку, запуск или обучение.

## 1. Принятые границы архитектуры

| Решение | Рабочий baseline M0 | Происхождение / статус |
|---|---|---|
| Система | Один Lion (GRU) + одно стационарное обучаемое поле W | Research Mission v0.2; сохранено |
| Поле | Тор `N=16`, `C=32`, `W:[4096,32]` FP32 | Миссия, C=32 — стартовая гипотеза, не оптимум |
| Адрес | Свободная **абсолютная** координата после каждого токена, не `p_prev+Δp` | Выбор владельца в PR #2; прежний bounded-step — будущий контроль |
| Голова | `u=h M_hᵀ + r_prev M_rᵀ + b_p`, период `N=16`; без прямой подачи `p_prev` | Выбор конкретного минимального engineering default из PR #5, §2–3 |
| Чтение | Одно новое дифференцируемое trilinear-read на токен (R=1), восемь углов и периодический wrap; sampler по **raw u**, detach только индексной ветви | Миссия + контракт PR #5 |
| Состояние | `h_t,p_t,r_t` — runtime state, не параметры; `r_0=R(W,p_0)` внутри loss closure; `h_0=0`, `p_0=(0.5,0.5,0.5)` | PR #5, обязательный путь градиента через r₀ |
| Контроллер | Reset-after GRU, `H=128`, embedding `d=32`, `V=32`; вход `[embedding(x_t);r_prev]` | Конкретный рабочий default PR #5, не измеренный optimum |
| Выход | `c=V_m h`, `m=U(r⊙c)`, `y=tanh(h+m+b_y)`, `logits=Dy+b_o`; сохраняется residual h | PR #5 §2; y не подменяет hidden state h |
| Обучение | Teacher forcing, полный BPTT, **один** differentiated loss по `B×T` позициям, **один** optimizer update; target входит только в loss/следующий teacher-forcing step | PR #2 разделяет per-token loss и update; PR #5 даёт точный контракт |
| Параметры | W и Lion обучаются из одного pre-update loss, один global clipping, затем две группы Adam; freeze W сохраняет coordinate gradients | PR #5; никаких независимых backward по группам |
| Inference | Параметры не меняются, входные токены обновляют h/p/r, при генерации выбранный токен подаётся обратно | PR #2; никакого online learning |
| Out of scope | Retrieval, журнал событий, несколько агентов, R>1, TBPTT, регуляризация покрытия, новые head architectures, UI-framework | PR #2/#5 и baseline: отдельные последующие эксперименты |

**Границы математической корректности:** corner ordering `000,001,010,011,100,101,110,111`; flatten `((x·16)+y)·16+z`; `λ≥0`, `Σλ≈1`; правильно обрабатываются отрицательные адреса и wrap; gradient по координатам проходит через `f=u−stopGradient(floor(u))`, индексам градиент не нужен. При сбое S01–S05/G01–G04 не обучать модель. Точный математический источник — [PR #5 / Mathematical and Interface Contracts](https://github.com/LL1yt/Gemina_project/blob/research/2026-11-02-detailed-mvp-plan/docs/plans/2026-11-02-mvp/MATHEMATICAL-INTERFACE-CONTRACTS.md).

## 2. Первые воспроизводимые настройки (зафиксированный кандидат, без tuning до измерений)

| Категория | Значение для первого пробного запуска |
|---|---|
| dtype, batch, target length | FP32; `B=32`, `T=64`, примеры `X:[B,T+1]` |
| Optimizer | Adam `β=(0.9,0.999)`, `ε=1e-8`, **biasCorrection=true**, weight decay=0 |
| LR | `θ=1e-3`, `W=1e-4`; global gradient clipping `c=1` |
| Regularizers/scheduler | dropout=0; без penalty/forced coverage; constant LR |
| Initialization | `W,E∼Normal(0,0.1²)`; GRU Wx/Wh `Uniform(−1/√H,+1/√H)`, biases=0; address head `Uniform(−s_p,+s_p)`, `s_p=√(6/(H+C+3))`, `b_p=(0.5,0.5,0.5)`; U/V_m/D по Xavier uniform, decoder biases=0 |
| Seeds | Initializations 11/22/33; train=1001, val=1002, IID test=1003, composition test=1004, shuffle=2001; независимые streams |
| Corpus (synthetic sanity) | `V=32`: 65 tokens = `BOS A_i B_b START` + 15 × `QUERY X_q ANSWER Y_{(2i+1)q+b mod8}` + `EOS`; split (i+b) mod4=0 held-out for composition |
| Proposed split sizes | train 1536, val 192, IID test 384, composition test 128; disjoint canonical example keys |
| Initial caps, **not authorization to execute** | tiny sanity: max 256 updates; first validation run: max 1000 updates, validate every 50; choose checkpoint via validation answer NLL, never test |
| Restricted LR fallback | If observed need and **separate capped trial**, `(1e-3,1e-3)` or `(3e-4,3e-4)` from PR #5. No silent tuning, no choosing on test. Head-scale=4 or σ_W=0.03 require their own reason and cap. |

All values in this section are a **selected, reproducible starting configuration** from the proposals, not empirically validated optima. A material change is a new explicit passport revision with reason; model selection and compute budgets are separate owner decisions.

## 3. Setup provenance / gates before M1

- Proposed dependency source: `mlx-swift 0.33.0`, commit `7ede4fd6d05f63305c1c0ebf4f8c1909a637c574` (PR #5 source audit). Actual package resolution, system Swift/Xcode, MLX/Metal linkage, numeric semantics and version compatibility remain **UNVERIFIED / NOT_RUN**. Swift manifest 6.3 is a proposal; if incompatible, stop and record choice, no silent downgrade.
- Use Swift/MLX on the owner's Mac only after separate authorization. Small SwiftPM core + runner/tests, no mandatory GUI, plugin framework, third-party tokenizer or dataset importer. If an Xcode project is later required, follow current [central Xcode project-format policy](https://github.com/LL1yt/Project_Index/blob/main/docs/XCODE-PROJECT-FORMAT.md) only after confirming current toolchain and file-format support.
- Before any M1 implementation, compare this passport with pinned PR #5 Contracts/Training/Test Matrix and AGENTS; record full SHA, platform, versions and parameters. The source PR #5 is still a **Draft**, not automatically merged into main. Do not copy its metadata conflicts into main wholesale.
- M1/M2 gates: analytical, independent fixtures first; sampler values S01–S05 and gradients G01–G04 must pass before claiming M3. M3 additionally checks the full causal gradient paths, especially W→r₀→GRU and gradients through u; full BPTT in one closure.
- **R1 diagnostic at M3:** before training and on fixed probe, record RMS(h), mean/std/range of raw u per axis, unique base cells `floor(u) mod16`, unique eight-corner nodes, and effective node count `N_eff=1/Σp_i²` from normalized accumulated read mass. Count reset read separately. Independent CPU NumPy checks by Morgana and Virgil suggest very low initial spread but do not establish Swift/MLX behaviour or learning failure. Low coverage alone is **not a failure gate**.
- M4 is a technical learning sanity, not an architecture competition. M5 snapshot/replay and M6 controls follow only when the relevant preceding gates pass. M6-A first GRU-only H128 plus field/address ablations; M6-B later capacity-match H248. R2/R3 additions are level 2, **not blockers** for working M1–M4.
- All runtime checks, builds, learning curves, checkpoints, Swift tests, CI and source execution in this repo remain **NOT_RUN** until performed and reported with provenance. The separate CPU NumPy checks are not Swift/MLX PASS.

## 4. Decision / conflict log

1. **Resolved for first reference:** PR #2 free absolute addressing takes precedence over the bounded-step formula in historical main; PR #5 affine + periodic raw-u sampler is its chosen concrete initial realisation. Preserve old form as historical proposal/control.
2. **Resolved for first reference:** full BPTT, R=1 and one optimizer update per B×T; independent losses are not automatically independent updates. A future local-credit experiment does not replace reference.
3. **Resolved for first reference:** one event record with eight weighted links from PR #2 is a **future** event-memory design and is not implemented in M0–M4. W remains trainable parameters, not runtime dialogue storage.
4. **Resolution mechanism:** exact Swift/MLX toolchain and budget remain a gated setup check, not an inference from documentation; a failing compatibility check calls for a logged M0 amendment.
5. **Deliberately deferred:** alternative head normalizations, expanded R2 controls, harder R3 tasks, LR/head scaling, second memory, parallel agents inside model. Implement only after data/explicit request.
6. **Source-of-truth:** this compact passport is the present choice for v0.1. Full derivations, detailed test identifiers and numerical tolerances remain in linked PR #5 until separately integrated; any divergence must be written as an explicit amendment. Existing PR #2 and #5 are not automatically merged or superseded as source histories.

**Next authorized action from this document alone:** review / reconcile setup choices; **no runtime authorization**, no silent merge of PR #2/#5, no launch.


## Последующее узкое поручение владельца — 2026-11-04

После утверждения паспорта владелец отдельно разрешил **Моргане реализовать только M1 sampler** с проверками S01–S05 в собственной ветке/Draft PR. **Вергилий** — независимые математические fixtures до просмотра реализации, затем аудит кода и результатов. Это **не конкурс**, не разрешение M2/M3/M4, полного обучения или изменения контрактов. До фактического исполнения все проверки остаются NOT_RUN; при недоступном Mac/toolchain исполнитель обязан указать BLOCKED. См. [AGENTS](../AGENTS.md) и [распределение ролей](AGENT-WORK-ALLOCATION-v0.1.md).
