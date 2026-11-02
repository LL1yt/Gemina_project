# Gemina — Test Matrix and Acceptance Criteria

**Дата:** 2026-11-02. **Статус:** DRAFT / PROPOSED TEST PLAN.  
**Все проверки прототипа в этой матрице: NOT_RUN.** Файлов тестов, fixtures модели и тестового корпуса в этой работе не создавалось. [Этапы](DETAILED-MVP-IMPLEMENTATION-PLAN.md) · [Контракты](MATHEMATICAL-INTERFACE-CONTRACTS.md) · [Обучение](TRAINING-EVALUATION-PROTOCOL.md).

Матрица разделяет исправность механизма и исследовательский вывод. Неудача в строгом инварианте останавливает следующий инженерный этап. Отсутствие преимущества над контролем — допустимый отрицательный или неопределённый исследовательский результат.

## 1. Предлагаемые численные допуски

Это **предварительные PROPOSED значения для утверждения перед реализацией**, не уже измеренная точность MLX.

| Проверка | Предлагаемый критерий |
|---|---|
| Integer IDs, shapes, keys, counts, masks | Точное совпадение |
| Значения sampler / суммы λ / independent-batch comparison | Элементная ошибка ≤10⁻⁵+10⁻⁴·max(abs(reference),abs(actual)), при fixtures масштаба порядка 1 |
| Аналитический gradient против autodiff sampler | ≤10⁻⁵+10⁻⁴·scale на невырожденной fixture |
| FP32 finite differences | ε из {10⁻²,3·10⁻³,10⁻³}; требовать устойчивого согласия хотя бы двух соседних ε: ошибка ≤5·10⁻⁴+2·10⁻²·scale |
| Same-model snapshot/replay logits/state | ≤10⁻⁵+10⁻⁴·scale; torus-distance для p |
| K=T / restored-next-update gradient и параметры | Начать с ≤10⁻⁵+10⁻⁴·scale; порядок reduction зафиксирован |
| Parameter fingerprint при inference / frozen W | Точные байты после eval; без допуска на изменение весов |
| Tiny sanity | Предложенный ориентир ≥95% train answer accuracy в cap, не исследовательский порог |
| Преимущество W/head | Универсальный процент не назначен; paired effects, seeds и границы вывода |

Scale=max(abs(a),abs(b)) в сравнении двух чисел. Отдельно сообщать maximum absolute error и normalized error; относительная ошибка около нуля не используется одна. Перед finite differences выбрать точки строго внутри ячеек и ε меньше расстояния до любой затрагиваемой грани. Для multi-step chain проверить все адреса обеих perturbations, чтобы случайное пересечение грани не маскировалось под ошибку производной.

Если tolerances не достигнуты, зафиксировать failing fixture и причину. Не ослаблять их автоматически до PASS. Backend variability, различный reduction order, неверный graph и ошибка формулы — разные объяснения.

## 2. Sampler и gradient

| ID / этап | Будущая проверка | PASS / причина FAIL | Статус |
|---|---|---|---|
| S01 / M1 | Логический [N,N,N,C] против flat; отдельные x/y/z/channel значения | Точные индексы и corner order; FAIL — оси/stride перепутаны | NOT_RUN |
| S02 / M1 | Углы, середина ячейки, случайные внутренние точки | λ≥0, Σλ≈1, read равен weighted reference; B=1 не теряет axis | NOT_RUN |
| S03 / M1 | u, u+16k по каждой оси; отрицательные и несколько периодов | Значения периодичны, indices 0…4095; FAIL — signed remainder/cast | NOT_RUN |
| S04 / M1 | Const W и отдельный channel ramp внутри выбранной ячейки | Const read; известная интерполяция ramp без неверного global periodic ramp assumption | NOT_RUN |
| S05 / M1 | Пара batch rows против отдельных вызовов; near-seam FP32 | Batch независим; p в [0,16), корректное округление endpoint | NOT_RUN |
| G01 / M2 | Scalar probe s=v·R(W,u), непостоянные corners и ненулевой v | AD по u совпадает с локальными разностями и finite differences вне граней | NOT_RUN |
| G02 / M2 | Производная s по выбранным W и одному непрочитанному узлу | λ·v для выбранных, ноль для непрочитанного в isolated sampler | NOT_RUN |
| G03 / M2 | Повторённые координаты/индексы, sum reduction | Gradient повторных обращений складывается; для mean отдельно учитывается нормировка | NOT_RUN |
| G04 / M2 | Точные грани, включая 0/16, и точки ±ε | Periodic equality и непрерывность значения; односторонние производные соответствуют своим ячейкам; central mismatch на изломе не FAIL | NOT_RUN |

Для G04 сравнивать R(−ε) с R(16−ε), R(+ε) с R(16+ε). Разность R(16−ε) и R(0) должна стремиться к нулю с ε с учётом локального наклона; она не обязана быть нулевой при конечном ε. У постоянного поля coordinate gradient должен быть нулём. Ненулевой gradient требуется только там, где выбранная fixture аналитически исключает вырождение.

## 3. Полный цикл и trainer

| ID / этап | Будущая проверка | PASS / причина FAIL | Статус |
|---|---|---|---|
| C01 / M3 | Встроенная GRU one-step против формулы, явный h₀ | [B,H], reset-after, верные gates и biases; FAIL — другое определение GRU | NOT_RUN |
| C02 / M3 | Parameter inventory и полная цепочка одного step | W/head/GRU/E/U/V_m/decoder присутствуют; state не является trainable leaf | NOT_RUN |
| C03 / M3 | Изменить будущие входы/targets при одинаковом доступном префиксе | Ранние logits не меняются; step не получает target или split label | NOT_RUN |
| C04 / M3 | Два шага с loss только второго; проверить r₀-path и ранний read/head | Совпадение выбранных multi-step derivatives; намеренный detach меняет проверяемый gradient, не forward | NOT_RUN |
| C05 / M3 | Один общий batch против независимых forward; reset two sessions | Нет переноса h/r между lanes/эпизодами; h_t не заменён y_t | NOT_RUN |
| O01 / M3 | Observer отключён/включён | Те же numerical outputs/gradients; summaries не удерживают graph и не читаются Lion | NOT_RUN |
| T01 / M4 | Shift inputs/targets, Int32, masks и пустая маска | 65→64; правильные answer indices; empty mask rejected; no loss после EOS/PAD в masked fixtures | NOT_RUN |
| T02 / M4 | Dataset keys и split manifest | Нулевое пересечение полных examples; held-out pairs отсутствуют в train; RNG namespaces и hashes зафиксированы | NOT_RUN |
| T03 / M4 | Один pre-update gradient и routing двух Adam | W только в своей группе, все trainable leaves покрыты единожды; biasCorrection=true; один update/counter обеих групп | NOT_RUN |
| T04 / M4 | Tiny sanity, затем свежий отчётный run | Конечные loss/gradients/parameters; обучаемые answer metrics улучшаются; cap и выбор checkpoint соблюдены | NOT_RUN |
| T05 / M4 | Два повторения с теми же seeds/версией/order и train→validation | Повторяемость в объявленном допуске; validation не меняет модель/optimizer и использует fresh state | NOT_RUN |

C04 должен отдельно проверить путь W→r₀→первый GRU и путь через последующее r. Проверка лишь одного прямого sampler-gradient не обнаружит случайное отсечение temporal paths. Для проверки graph не требуется большое обучение.

T04 не требует total NLL→0: случайные query-токены дают irreducible loss. Если модель численно исправна, но не достигает предложенного tiny threshold, это диагностический FAIL данного прогона / LEARNING_NOT_ESTABLISHED; сначала проверить loss shift, scales, routing и caps. Не объявлять всю архитектуру невозможной.

## 4. Inference, checkpoint и replay

| ID / этап | Будущая проверка | PASS / причина FAIL | Статус |
|---|---|---|---|
| R01 / M5 | Prompt, greedy continuation, max-token/EOS stop | По одному step на потреблённый token; θ/W fingerprints не меняются | NOT_RUN |
| R02 / M5 | Model+optimizer round-trip после completed update | Exact keys/shapes/dtypes/bytes; следующий batch/update соответствует uninterrupted run в допуске | NOT_RUN |
| R03 / M5 | Snapshot после consumed token, до sampling; продолжение | State/logits согласованы; нет повторного consumption и скрытого extra-read | NOT_RUN |
| R04 / M5 | Replay одного полного префикса с тем же reset/model | h/r/logits и torus p соответствуют; shortened prefix не выдаётся за восстановление всей истории | NOT_RUN |
| R05 / M5 | Другие θ/W/config/vocabulary, missing buffer, corrupt metadata | Strict rejection; пересчёт r не маскирует несовместимый h | NOT_RUN |

У greedy tie-break — меньший token ID. Если numerical differences меняют argmax при практически равных logits, фиксировать margin и неоднозначность; не обещать точную одинаковую строку только из tolerance по logits. Cross-device/cross-version bitwise reproducibility не предполагается.

M5 проверяет правильность механизмов генерации и состояния, а не связный естественный текст. Предлагаемый предел первого диагностического continuation — 64 новых токена или EOS; некорректные/reserved выходы не скрываются фильтром, а отражаются в отчёте.

## 5. Контрольные сравнения и сила результата

| ID / этап | Будущая проверка | Критерий корректного протокола | Статус |
|---|---|---|---|
| E01 / M6 | GRU-only и frozen random W | Оба пути r удалены только у no-W; frozen W сохраняет coordinate gradient; training from scratch | NOT_RUN |
| E02 / M6 | Static и fixed addressing | Head отсутствует, число чтений согласовано; путь не зависит от target/token content | NOT_RUN |
| E03 / M6 | Capacity GRU H=248 | Actual count совпадает с контрактом; явно указано отличие параметров и вычислений | NOT_RUN |
| E04 / M6 | Paired seeds, exposures, selection и interventions | Одни splits/order/caps; no test tuning; effects и разброс; perturbation не заменяет scratch controls | NOT_RUN |
| E05 / M6-A → M6-B | Очерёдность: GRU-only H128 первым; GRU H248 вторым | Зафиксировать первичные paired metrics и границы вывода M6-A; затем выполнить capacity control M6-B, не называя первый результат проверкой равных параметров | NOT_RUN |

**Порядок:** M6-A (Gemina против GRU-only H=128, затем причинные ablations W/head); M6-B (GRU-only H=248 с близким общим числом параметров). Для фиксированных допусков и independent sampler-fixtures двух агентов до обучения см. [workflow](INDEPENDENT-SAMPLER-TESTS-AND-AGENT-WORKFLOW.md). Результат M6-A не достаточен для заявления о преимуществе при равной параметрической ёмкости.

Результат E01–E05 — корректно проведённое сравнение, а не требование заранее обеспечить победу Gemina. Порог технического завершения и сила исследовательского вывода сообщаются отдельно:

| Уровень | Что нужно для положительного вывода | Что остаётся неизвестным |
|---|---|---|
| Технически исправный MVP | M1–M5 проходят объявленные gates | Полезность обучения W |
| Фактический вклад W | Измеримое влияние W на held-out outputs и реальные learning updates; причинные контроли | Может ли ту же функцию дать random/static field |
| Польза обучаемого W | Преимущество над frozen random W с устойчивым направлением paired effects | Польза выбора адресов |
| Польза head | Преимущество над static/fixed | Уникальная ценность 3D topology |
| Контроль дополнительной ёмкости | Сопоставление с близкой по параметрам GRU | Превосходство по compute, языку, длинному контексту и общность результатов |

Положительная динамика loss без W-contribution означает работающий predictor, но не подтверждённую центральную гипотезу Gemina. Отсутствие превосходства над обычной GRU не отменяет инженерный результат. Равные результаты на лёгкой задаче, противоречивые seeds и недостаточный budget обозначаются INCONCLUSIVE.

## 6. Только последующие опыты

| ID / этап | Проверка | PASS / интерпретация | Статус |
|---|---|---|---|
| F01 / M7 | K=128 против full BPTT T=128 | Loss/gradients/update совпадают в допуске, одинаковый M и один update | NOT_RUN |
| F02 / M7 | K=8/16/32/64 | Forward при fixed weights совпадает; градиент усечён именно по h/r; chunk graphs освобождаются; измерения памяти отдельно | NOT_RUN |
| F03 / M7 | D=8/32/96, paired no-cue/control | Точный offset, ни одного промежуточного cue-dependent target; общий M=128; independent group splits | NOT_RUN |
| F04 / future | R=2/4 | Требуется отдельный transition/loss и budget plan; не gate первого MVP | NOT_RUN |

Численное сохранение h/r через K не означает сохранения графа. Совпадение scalar loss при TBPTT не означает совпадения gradient. При более коротком K качество может снизиться; это измеряемый компромисс, а не заранее ожидаемый PASS.

## 7. Что реально проверено при подготовке документации

Прочитаны актуальные исходные документы и PR heads, сопоставлены математические формы, API и статусы решений; выполнена независимая содержательная проверка. Проверки относительных ссылок, состава Markdown diff и GitHub read-back относятся только к документации и фиксируются в PR/PROVENANCE.

Эти действия не являются выполнением S/G/C/T/R/E/F. Ни исторические числа из DOCX, ни чтение upstream исходников, ни публикация Draft PR не переводят runtime-статусы из NOT_RUN в PASS.

