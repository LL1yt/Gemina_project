# Literature / research notes

**Research / Architecture Draft.** Перенесённое изложение §6 исходного [DOCX](../sources/Gemina_Architecture_Draft_v0_1.docx); номера источников сохранены. Это архивирование выполненного ранее обзора, не новый систематический поиск литературы. Указанная в оригинале дата обращения — 2026-11-02; ссылки извлечены из DOCX, внешние страницы в этой задаче повторно не проверялись. При проектировании реализации проверить актуальные API и закрепить версии.

Обзор сопоставляет механизмы, а не устанавливает приоритет изобретения. Самая близкая опорная комбинация — непрерывная головка LANTM, обучаемые пространственные векторы Instant NGP и параметрическая память PKM. В таблице сравнения ниже нет основания считать Gemina доказанно новой либо считать результаты этих работ результатами Gemina.

## 6 1 Навигация и пространственное чтение

Lie Access Neural Turing Machines, Yang и Rush, 2017. Контроллер двигает непрерывную головку на многообразии адресов через действия группы; чтение использует сглаживание. Полезны относительные перемещения, сохранение адреса между шагами и обучение маршрута через BPTT. Важное отличие: содержимое памяти записывается во время последовательности, а опубликованный механизм допускает и абсолютную адресацию. Gemina ограничивает движение локальным шагом по фиксированному тору и использует межэпизодические параметры W. Работа демонстрирует конкретные алгоритмические задачи, но не решает вопрос полезности такого W для языкового предсказания. [1]

Spatial Transformer Networks, Jaderberg и соавторы, 2015. Нейросеть предсказывает координаты sampling; интерполяция пропускает градиент к входной карте и координатам. В статье выведены производные, приложение рассматривает объёмный случай. Можно переиспользовать математическую конструкцию sampler. Отличие Gemina — чтение параметров модели и перенос позиции между токенами, тогда как STN обычно преобразует feature map входа. Наличие корректного sampler не решает проблему исследования удалённых областей. [5]

Instant Neural Graphics Primitives, Müller и соавторы, 2022. Маленькая сеть обучается вместе с пространственными grid features, которые интерполируются по координатам. Это сильный предшественник статического обучаемого поля. Можно перенести совместный градиент и позже несколько разрешений. Здесь координаты задаются графической задачей; у Gemina их выбирает рекуррентный агент. Hash encoding и CUDA оптимизации не нужны для первого поля 16³, а опубликованные скорости не являются прогнозом для MLX. [6]

## 6 2 Память контроллеров и постоянные параметры

Neural Turing Machines, Graves и соавторы, 2014. Рекуррентный контроллер обучается читать и записывать внешний массив через дифференцируемые веса адресации. Полезны разделение контроллера и access module, взвешенное чтение и обучение через время. В NTM содержимое памяти меняется в эпизоде; W Gemina обновляется между обучающими проходами. Content addressing, erase/add и вся схема tape addressing не требуются для MVP. [2]

Differentiable Neural Computer, Graves и соавторы, 2016. Дополняет память учётом занятости, освобождением, выделением места и временными связями записей. Эти механизмы решают конкретные проблемы runtime памяти и могут пригодиться при будущем режиме записи. Gemina имеет постоянную геометрию и не строит temporal links в первом прототипе. Официальная реализация ясно отделяет controller, access, TemporalLinkage и Freeness. [3]

Dynamic Neural Turing Machine, Gulcehre и соавторы, 2016–2017. Ячейки разделены на обучаемые адресные векторы и эпизодическое содержимое; рассмотрен GRU контроллер. Особенно полезно различение параметров и оперативной памяти. D NTM обучает адреса, а Gemina фиксирует координаты и обучает значения поля. Его эксперименты не устанавливают преимущество трёхмерного тора. [4]

Large Memory Layers with Product Keys, Lample и соавторы, 2019. Память содержит обучаемые keys и values, общие для примеров; query выбирает малое число параметров. Это близко к идее распределить ёмкость между доступом и адресуемыми значениями. Отличие — поиск по ключам вместо непрерывного локального маршрута. Полезны измерение взвешенного использования и отдельная настройка LR памяти. Исследовательский вопрос Gemina — эффект геометрии и траектории поверх параметрической ёмкости. [7]

Augmenting Self attention with Persistent Memory, Sukhbaatar и соавторы, 2019. Постоянные обучаемые векторы дополняют attention и могут выполнять роль вычислительного слоя. Это подтверждает, что память бывает параметрической, а не только хранилищем эпизода. Gemina отличается рекуррентной адресацией и локальным движением. Механизм самой статьи остаётся Transformer архитектурой и не переносится в Gemina целиком. [8]

## 6 3 Быстрые веса адаптация и самоорганизация

Learning to Control Fast Weight Memories, Schmidhuber, 1992. Одна сеть обучается выдавать контекстные изменения весов другой. Using Fast Weights to Attend to the Recent Past, Ba и соавторы, 2016, использует быстро меняющуюся ассоциативную матрицу для недавнего контекста. Полезно разделение медленных параметров, активностей и быстрых следов. У Gemina v0.1 оба набора параметров фиксированы при inference. Разные offline LR ещё не делают её fast weight системой; эти работы относятся к будущей записи. [9, 10]

Titans Learning to Memorize at Test Time, Behrouz и соавторы, 2025. Нейронная memory module обновляется по контексту с внутренней ошибкой, momentum и forgetting; отдельно выделяется persistent memory. Можно использовать это разграничение для будущего online режима. Ключевое отличие: Gemina v0.1 не выполняет test time updates. Replay prompt восстанавливает временное состояние и не является аналогом обучения memory module Titans. [11]

Growing Neural Cellular Automata, Mordvintsev и соавторы, 2020. Общее обучаемое правило обновляет состояния клеток через локальное окружение; в опытах получаются рост, устойчивость и регенерация изображений. Это конкретный пример распределённого исполнения функции. Состояния клеток не равны независимым медленным параметрам каждой координаты; обучение использует общую ошибку. Для Gemina полезна постановка испытаний локального взаимодействия, но независимая кооперация разнородных агентов остаётся гипотезой. [12]

HyperNetworks, Ha, Dai и Le, 2016–2017. Одна сеть генерирует параметры другой; полезна связь с условным преобразованием hidden state. В Gemina предложена более простая модуляция низкого ранга коэффициентами читаемого поля. Это механизм для проверки разделения вычислений, а не основание утверждать новую категорию сетей. [13]

## 6 4 Что остаётся самостоятельной гипотезой

Контекстная локальная траектория и медленное обучаемое поле смогут совместно сформировать полезный предиктор при компактной GRU.

Ограниченная геометрия создаст полезную организацию параметров вместо случайной таблицы, неподвижного bias или обхода поля.

Разница темпов обучения позволит устойчиво разделить адаптацию агента и организацию поля.

В дальнейшем разные агенты смогут обнаруживать переносимые соответствия между самостоятельно сформированными структурами.

Ни один из этих результатов не следует автоматически из изученных работ. На первом этапе проверяется первая гипотеза и собирается минимальная диагностика второй; дальнейшие выводы зависят от результатов.

## Библиография исходного исследования

[1] Yang G, Rush A M. Lie Access Neural Turing Machines. ICLR 2017. [Источник](https://arxiv.org/abs/1611.02854)

[2] Graves A, Wayne G, Danihelka I. Neural Turing Machines. 2014. [Источник](https://arxiv.org/abs/1410.5401)

[3] Graves A et al. Differentiable Neural Computers. Авторское описание и официальная реализация. 2016. [Источник](https://deepmind.google/blog/differentiable-neural-computers/)

[3a] Google DeepMind. DNC source repository. [Источник](https://github.com/google-deepmind/dnc)

[4] Gulcehre C et al. Dynamic Neural Turing Machine with Soft and Hard Addressing Schemes. 2016–2017. [Источник](https://arxiv.org/abs/1607.00036)

[5] Jaderberg M et al. Spatial Transformer Networks. NeurIPS 2015. [Источник](https://arxiv.org/abs/1506.02025)

[6] Müller T et al. Instant Neural Graphics Primitives with a Multiresolution Hash Encoding. SIGGRAPH 2022. [Источник](https://nvlabs.github.io/instant-ngp/)

[7] Lample G et al. Large Memory Layers with Product Keys. NeurIPS 2019. Особенно разделы 4 2 и 4 3. [Источник](https://arxiv.org/abs/1907.05242)

[8] Sukhbaatar S et al. Augmenting Self attention with Persistent Memory. 2019. [Источник](https://arxiv.org/abs/1907.01470)

[9] Schmidhuber J. Learning to Control Fast Weight Memories. Neural Computation 1992. [Источник](https://people.idsia.ch/~juergen/fastweights/)

[10] Ba J et al. Using Fast Weights to Attend to the Recent Past. NeurIPS 2016. [Источник](https://arxiv.org/abs/1610.06258)

[11] Behrouz A, Zhong P, Mirrokni V. Titans Learning to Memorize at Test Time. 2025. [Источник](https://arxiv.org/abs/2501.00663)

[12] Mordvintsev A et al. Growing Neural Cellular Automata. Distill 2020. [Источник](https://distill.pub/2020/growing-ca/)

[13] Ha D, Dai A, Le Q V. HyperNetworks. 2016–2017. [Источник](https://arxiv.org/abs/1609.09106)

[14] Lian X et al. Can Decentralized Algorithms Outperform Centralized Algorithms. NeurIPS 2017. [Источник](https://arxiv.org/abs/1705.09056)

[15] MLX Swift. Module и документация нейросетевых модулей. [Источник](https://github.com/ml-explore/mlx-swift/blob/main/Source/MLXNN/Module.swift)

[15a] MLX Swift 0.33.0. Recurrent.swift, проверенная реализация GRU. [Источник](https://github.com/ml-explore/mlx-swift/blob/0.33.0/Source/MLXNN/Recurrent.swift)

[16] MLX Swift. Optimizers.swift, включая MultiOptimizer и Adam. [Источник](https://github.com/ml-explore/mlx-swift/blob/main/Source/MLXOptimizers/Optimizers.swift)

[17] MLX Swift. Официальная таблица API Python и Swift. [Источник](https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/Documentation.docc/Articles/converting-python.md)

[18] MLX Swift. Релизы и изменения API. [Источник](https://github.com/ml-explore/mlx-swift/releases)

[19] Kornblith S et al. Similarity of Neural Network Representations Revisited. ICML 2019. [Источник](https://proceedings.mlr.press/v97/kornblith19a.html)

Новизна Gemina и полезность сочетания механизмов не установлены. Результаты цитируемых работ не являются результатами Gemina. См. также [research notes](RESEARCH-NOTES.md).
