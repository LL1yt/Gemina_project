# Математическая спецификация кандидата

**DRAFT / proposed; не реализация и не принятый паспорт запуска.** Переложение §2 [Architecture Draft v0.1](../sources/Gemina_Architecture_Draft_v0_1.docx). Формулы записаны в LaTeX с сохранением смысла; при расхождении обратиться к оригиналу и явно исправить производный документ.

> **Смена предпочтительного навигатора (2026-11-02):** формула `Δp=s·tanh(...)` в этом документе сохранена как **исторический ограниченный кандидат**. Для первого будущего прототипа пользователь выбрал **свободное непрерывное абсолютное адресование**, но новую конкретную функцию p_t ещё предстоит выбрать и проверить. Не считать s=0.5 действующим решением. [Уточнение](proposals/2026-11-02-ARCHITECTURE-CLARIFICATIONS.md).

## Формы и начальные условия

Пусть K — размер словаря, dₑ=32, H=128, C=32, N=16. dₑ/H и конкретные начальные условия предложены черновиком.

$$W\in\mathbb R^{N\times N\times N\times C},\quad E\in\mathbb R^{K\times d_e},\quad h_t\in\mathbb R^H,\quad p_t\in[0,N)^3,\quad r_t\in\mathbb R^C.$$

$$h_0=0,\qquad p_0=(0.5,0.5,0.5),\qquad r_0=R(W,p_0).$$

θ объединяет все параметры кроме W. Индекс t относится к токену, n — к обновлению оптимизатора.

## GRU

$$e_t=E[x_t],\qquad a_t=[e_t;r_{t-1}],\qquad h_t=\mathrm{GRU}_{\theta_G}(a_t,h_{t-1}).$$

Конкретный предложенный reset-after вариант (g — reset gate, r — только чтение):

$$z_t=\sigma(A_z a_t+B_z h_{t-1}+b_z),\qquad g_t=\sigma(A_g a_t+B_g h_{t-1}+b_g),$$
$$\widetilde h_t=\tanh\!\left(A_h a_t+b_h+g_t\odot(B_h h_{t-1}+c_h)\right),$$
$$h_t=(1-z_t)\odot\widetilde h_t+z_t\odot h_{t-1}.$$

A имеет форму H×(dₑ+C), B — H×H, bias — H. Соответствие конкретной версии MLX должно быть заново проверено перед реализацией; указание версии 0.33.0 в исходнике не является установленной зависимостью этого репозитория.

## Навигация

$$v_t=M_hh_t+M_rr_{t-1}+b_p,\qquad \Delta p_t=s\tanh(v_t),\qquad s=0.5,$$
$$u_t=p_{t-1}+\Delta p_t,\qquad p_t=u_t-N\lfloor u_t/N\rfloor.$$

Mₕ: 3×H, Mᵣ: 3×C, bₚ: 3. Операции покоординатны. Ограничение шага относится к каждой оси; евклидова длина ограничена √3/2. Это трёхмерный тор, а не поверхность двумерного тора.

## Периодическое трилинейное чтение

$$i=\lfloor p_t\rfloor,\qquad f=p_t-i,\qquad q_b=(i+b)\bmod N,\qquad b\in\{0,1\}^3,$$
$$\lambda_b(f)=\prod_{d=1}^{3}\left[b_df_d+(1-b_d)(1-f_d)\right],$$
$$r_t=R(W,p_t)=\sum_{b\in\{0,1\}^3}\lambda_b(f)W_{q_b}.$$

λ неотрицательны и суммируются в 1. Чтение использует 8×32 значений и возвращает C=32. Выбор индекса не дифференцируется. Для координатного пути используются f=p−stopGradient(floor(p)) и wrap(u)=u−N·stopGradient(floor(u/N)). Это производная почти всюду; на гранях ячеек гладкость не гарантируется.

$$\frac{\partial r_{t,c}}{\partial W_{q_b,c'}}=\lambda_b(f)\,\mathbf 1_{c=c'},\qquad
\frac{\partial r_{t,c}}{\partial p_{t,d}}=\sum_b\frac{\partial\lambda_b(f)}{\partial f_d}W_{q_b,c}.$$

При повторном использовании узлов вклады градиента суммируются. Нельзя отсоединять всё r или p внутри обучающего фрагмента. Если соседние значения W одинаковы, координатный сигнал исчезает; исходник предлагает малую случайную инициализацию, например σ=0.1, а не константу.

## Модуляция и выход

$$m_t=U\left(r_t\odot(Vh_t)\right),\qquad y_t=\tanh(h_t+m_t+b_y),$$
$$F(r_t)=I+U\operatorname{diag}(r_t)V,\qquad y_t=\tanh(F(r_t)h_t+b_y),$$
$$\ell_t=Dy_t+b_o,\qquad \pi_t=\operatorname{softmax}(\ell_t),\qquad \pi_t(k)=P(x_{t+1}=k\mid x_{\le t}).$$

U: H×C; V: C×H; D: K×H; bᵧ: H; bₒ: K. Матрицу F явно строить не требуется. Residual h сохраняется. Одновременная нулевая инициализация W/U/V может погасить полезный градиент и не предлагается.

## Loss и update

Для непустого набора целевых позиций с маской μ (в исходнике обозначена m):

$$L_{pred}=-\frac{\sum_{b,t}\mu_{bt}\log\pi_{bt}(x_{b,t+1})}{\sum_{b,t}\mu_{bt}}.$$

Padding исключён; xₜ₊₁ не доступен forward текущего шага. μ переименована только для отличия от вектора модуляции mₜ.

$$g_\theta=\nabla_\theta L_{pred},\qquad g_W=\nabla_W L_{pred},$$
$$\alpha=\min\left(1,\frac{c}{\sqrt{\lVert g_\theta\rVert_2^2+\lVert g_W\rVert_2^2}}\right),\qquad (g_\theta,g_W)\leftarrow\alpha(g_\theta,g_W).$$

При нулевой общей норме α=1. Это явная запись предложенного глобального clipping, c=1 — настраиваемый кандидат.

$$ (\theta^{n+1},o_A^{n+1})=\mathrm{AdamStep}(\theta^n,g_\theta,o_A^n;\eta_A),$$
$$ (W^{n+1},o_W^{n+1})=\mathrm{AdamStep}(W^n,g_W,o_W^n;\eta_W).$$

Оба градиента получены от одного состояния модели до обновлений. Два состояния Adam; bias correction предлагается включить явно, weight decay выключить. β/ε, batch size, число обновлений и допуски не закреплены; это не исполнимый протокол. Подробности — [training design](TRAINING-DESIGN.md).
