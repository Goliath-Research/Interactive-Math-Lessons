import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import bernoulli, binom

    return bernoulli, binom, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # From One Trial to a Binomial Count

    ## Learning goals
    - Identify a discrete random variable and its possible values.
    - Explain the assumptions behind Bernoulli and binomial models.
    - Distinguish theoretical probabilities from simulated relative frequencies.
    - Connect a binomial count to a sum of independent Bernoulli trials.
    - Calculate exact, cumulative, upper-tail, and interval probabilities.
    - Compare theoretical means and variances with observed summaries.

    ## Opening challenge
    You have ten attempts at a target. Each attempt succeeds with probability 0.8,
    independently of the others.

    **Which is more likely: exactly eight successes, or eight or more successes?**
        - Make a prediction and explain it before calculating. 
        - Then explore how single yes-or-no outcomes become a count. 
        - We will return to this challenge below.

    ## 1. What is a discrete random variable?
    A **random variable** assigns a number to an experiment's outcome.

    A **discrete** random variable has a finite or countably infinite set of possible values.

    For example, let X be the face of a fair die: X can be 1, 2, 3, 4, 5, or 6.
    Its probability distribution gives P(X = k) = 1/6 for each of these values.
    Every probability is nonnegative, and all probabilities add to one.

    A **probability mass function (PMF)** gives P(X = k) for each possible value k.
    A simulated sample gives observed counts and relative frequencies. Those frequencies need not match the PMF exactly.

    | Model | Meaning of X | Possible values | Parameters |
    |---|---|---|---|
    | Bernoulli | Success indicator for one trial | 0, 1 | p |
    | Binomial | Success count in n trials | 0 through n | n, p |

    “Success” simply means the outcome being counted. It need not be desirable.
    All datasets below are simulated; no external files are needed.

    ### Three probability functions
    For both distributions, SciPy uses the same methods:

    - **Probability mass function (PMF):** P(X = k), calculated with **.pmf(k)**.
    - **Cumulative distribution function (CDF):** P(X ≤ k), calculated with **.cdf(k)**.
    - **Survival function (SF):** P(X > k), calculated with **.sf(k)**.

    SF(k) = 1 − CDF(k).
    - For integer-valued X, “at least k” is SF(k − 1).
    - For integers a ≤ b, P(a ≤ X ≤ b) = CDF(b) − CDF(a − 1).

    ## Reproducible experiments
    A seed is a number that lets us repeat a simulation and obtain the same results.
    Each simulation uses a fixed seed derived from the simulation seed control, so running other cells does not affect its results.
    Changing the seed can produce different observations, but the distribution’s probabilities
    remain the same.

    To compare sample sizes, we generate one large sample and examine its first 10 observations,
    then its first 100, and then its first 1,000. Each larger sample includes the observations
    already examined, showing how the results change as we include more observations.
    """)
    return


@app.cell
def _(mo):
    seed_control = mo.ui.slider(start=1, stop=100, step=1, value=42, show_value=True, label="Simulation seed")
    seed_control
    return (seed_control,)


@app.cell
def _(np, pd, plt):
    def draw_sample(model, size, seed):
        """Generate a reproducible sample."""
        return model.rvs(size=size, random_state=seed)


    def compare_summaries(values, model):
        """Compare model properties with descriptive summaries of the sample."""
        return pd.DataFrame({
            "Quantity": ["Mean", "Variance", "Standard deviation"],
            "Theoretical": [model.mean(), model.var(), model.std()],
            "Observed": [values.mean(), values.var(ddof=0), values.std(ddof=0)]
        })


    def plot_distribution(values, model, support, title):
        """Compare relative frequencies with probability masses at integers."""
        support = np.asarray(support)
        counts = np.bincount(np.asarray(values, dtype=int), minlength=int(support[-1]) + 1)
        observed = counts[support] / len(values)
        theoretical = model.pmf(support)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(support, observed, width=0.75, color="tab:blue", alpha=0.6, label="Observed relative frequency")
        ax.plot(support, theoretical, "o", color="tab:orange", label="Theoretical probability")
        ax.set(xlabel="Value of X", ylabel="Probability / relative frequency", title=title)
        ax.set_xticks(support[::max(1, len(support) // 16)])
        ax.set_xlim(-0.6, support[-1] + 0.6)
        ax.set_ylim(0, max(0.05, observed.max(), theoretical.max()) * 1.20)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        plt.close(fig)
        return fig

    return compare_summaries, draw_sample, plot_distribution


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Bernoulli: one yes-or-no outcome
    Let X = 1 for success and X = 0 otherwise. Then:

    $$P(X=1)=p,\qquad P(X=0)=1-p.$$

    For a fair coin, define heads as success and set p = 0.5. A weighted coin can use another p.
    The theoretical mean is p; the theoretical variance is p(1−p).
    For a sample of 0s and 1s, its mean is exactly its observed proportion of successes.

    **Predict first:** With p = 0.8, how many successes do you expect in 100 trials? Must the observed count equal that prediction?
    """)
    return


@app.cell
def _(mo):
    bern_p = mo.ui.slider(start=0, stop=1, step=0.05, value=0.8, show_value=True, label="Bernoulli probability p")
    bern_size = mo.ui.slider(steps=[10, 30, 100, 1000, 10000], value=100, show_value=True, label="Bernoulli observations")
    mo.hstack([bern_p, bern_size])
    return bern_p, bern_size


@app.cell
def _(bern_p, bernoulli, draw_sample, seed_control):
    bern_model = bernoulli(p=bern_p.value)
    bern_sequence = draw_sample(bern_model, 10000, seed_control.value)
    return bern_model, bern_sequence


@app.cell
def _(bern_model, bern_sequence, bern_size, mo):
    bern_values = bern_sequence[:bern_size.value]
    mo.md(f"**Observed successes:** {int(bern_values.sum())} of {len(bern_values)}  \n**Observed proportion:** {bern_values.mean():.4f}  \n**Theoretical success probability:** {bern_model.pmf(1):.4f}")
    return (bern_values,)


@app.cell
def _(bern_model, bern_values, plot_distribution):
    plot_distribution(bern_values, bern_model, [0, 1], "Bernoulli: probability model and observations")
    return


@app.cell
def _(bern_model, bern_values, compare_summaries, mo):
    mo.Html(
        compare_summaries(bern_values, bern_model).round(4).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell
def _(bern_values, np, plt):
    _shown = bern_values[:30]
    _figure, _axes = plt.subplots(figsize=(9, 4))
    _axes.plot(np.arange(1, len(_shown) + 1), _shown, "o", color="tab:blue")
    _axes.set(xlabel="Trial number", ylabel="Outcome", yticks=[0, 1], ylim=(-0.2, 1.2), title="First trials: this is a sequence, not a distribution plot")
    _figure.tight_layout()
    plt.close(_figure)
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Compare 10, 100, and 10,000 observations with p fixed at 0.8. Is agreement guaranteed to improve at every increase?
    2. Try p = 0 and p = 1. What happens to the outcomes and variance?
    3. Hold the parameters fixed and change the seed. Which quantities change: observed summaries, theoretical summaries, or both?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    1. No. Agreement need not improve at every increase; relative frequencies fluctuate.
    2. At p = 0, every outcome is 0. At p = 1, every outcome is 1. In either case the variance is zero.
    3. Observed summaries can change; theoretical summaries remain unchanged. At p = 0 or 1, the observations are constant regardless of seed.

    **Discussion:** The expected success count in 100 trials at p = 0.8 is 80, but it is not guaranteed.
    Larger samples generally approximate model probabilities more closely, with random fluctuations.
    At p = 0 or 1 the outcome is constant and the variance is zero.
        """)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. From Bernoulli trials to a binomial count
    Let B₁, …, Bₙ be independent Bernoulli trials with the same success probability p.
    Their sum X = B₁ + ⋯ + Bₙ counts successes and follows a binomial distribution.

    The next display shows **five experiments**. Start with ten trials per experiment, then change the controls. Each row sum is one binomial observation. Predict how the totals will change before moving a slider.
    """)
    return


@app.cell
def _(mo):
    bridge_n = mo.ui.slider(start=1, stop=20, step=1, value=10, show_value=True, label="Attempts in each experiment")
    bridge_p = mo.ui.slider(start=0, stop=1, step=0.05, value=0.8, show_value=True, label="Success probability")
    mo.vstack([bridge_n, bridge_p])
    return bridge_n, bridge_p


@app.cell
def _(bridge_n, bridge_p, mo, np, pd, plt, seed_control):
    _rng = np.random.default_rng(seed_control.value + 1)
    # Fixed uniforms make the change in p easy to follow visually.
    _uniforms = _rng.random((5, 20))
    _trials = (_uniforms[:, :bridge_n.value] < bridge_p.value).astype(int)
    _fig, _ax = plt.subplots(figsize=(9, 4))
    from matplotlib.colors import ListedColormap
    _ax.imshow(_trials, cmap=ListedColormap(["#e5e7eb", "#2563eb"]), vmin=0, vmax=1, aspect="auto")
    for _row in range(5):
        for _col in range(bridge_n.value):
            _ax.text(_col, _row, str(_trials[_row, _col]), ha="center", va="center",
                     color="white" if _trials[_row, _col] else "#374151")
    _ax.set_xticks(range(bridge_n.value), range(1, bridge_n.value + 1))
    _ax.set_yticks(range(5), [f"Experiment {i}" for i in range(1, 6)])
    _ax.set(xlabel="Attempt number", title="Five experiments: blue 1 = success, gray 0 = failure")
    _fig.tight_layout()
    plt.close(_fig)
    _totals = pd.DataFrame({"Experiment": range(1, 6), "Success count X": _trials.sum(axis=1)})
    mo.vstack([_fig, mo.Html(_totals.to_html(index=False, border=0).replace("<table ", '<table style="width: auto;" ')),
        mo.md(f"Each cell is one **Bernoulli outcome**. Each row total is one **Binomial count**, with n = {bridge_n.value} and p = {bridge_p.value:.2f}.\n\n**Try it yourself:** Increase p while holding n fixed. Which cells can change? Now increase n. Are you changing the number of experiments, or the attempts within each experiment?")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md("Increasing p can turn failures into successes in this display because the same underlying random numbers are reused. Increasing n adds attempts to each row; there are still five experiments. This shared-random-number display helps compare settings; it does not mean real experiments must change in this coupled way.")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Binomial: successes in a fixed number of trials
    The model requires a fixed trial count n, two outcomes per trial, independence, and the same p for all trials.
    Its possible values are 0 through n.

    $$P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}.$$

    The coefficient counts the ways to position k successes among n trials.
    The theoretical mean is np and the variance is np(1−p).

    In `binom.rvs(n=10, p=0.5, size=1000)`, **n** is trials per experiment and **size** is the number of repeated experiments. Each returned number is a count, not a proportion.
    """)
    return


@app.cell
def _(mo):
    bin_n = mo.ui.slider(start=1, stop=30, step=1, value=10, show_value=True, label="Trials per experiment n")
    bin_p = mo.ui.slider(start=0, stop=1, step=0.05, value=0.5, show_value=True, label="Success probability p")
    bin_size = mo.ui.slider(steps=[10, 100, 1000, 10000], value=1000, show_value=True, label="Repeated experiments displayed")
    mo.vstack([bin_n, bin_p, bin_size])
    return bin_n, bin_p, bin_size


@app.cell
def _(bin_n, bin_p, binom, draw_sample, seed_control):
    bin_model = binom(n=bin_n.value, p=bin_p.value)
    bin_sequence = draw_sample(bin_model, 10000, seed_control.value + 2)
    return bin_model, bin_sequence


@app.cell
def _(bin_sequence, bin_size):
    bin_values = bin_sequence[:bin_size.value]
    # The chart below summarizes these experiment counts.
    return (bin_values,)


@app.cell
def _(bin_model, bin_n, bin_values, np, plot_distribution):
    plot_distribution(bin_values, bin_model, np.arange(bin_n.value + 1), "Binomial: success counts across repeated experiments")
    return


@app.cell
def _(bin_model, bin_values, compare_summaries, mo):
    mo.Html(
        compare_summaries(bin_values, bin_model).round(4).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Shape and skewness
    - For 0 < p < 1, the theoretical binomial distribution is symmetric when p = 0.5.
    - For p < 0.5 it is **right-skewed**: most mass is at smaller counts, with a tail toward larger counts.
    - For p > 0.5 it is **left-skewed**: most mass is at larger counts, with a tail toward smaller counts.

    Skewness is named for the tail direction, not where the tallest bars stand.

    At p = 0 or 1, the distribution is concentrated at one value.

    A simulated chart need not reproduce theoretical symmetry exactly.

    ### Try it yourself
    1. Set n = 10 and compare p = 0.2, 0.5, and 0.8. Predict each theoretical mean and tail direction first.
    2. With n and p fixed, increase the repeated experiment count. Does the theoretical distribution change?
    3. With p = 0.5, increase n from 10 to 20. What happens to the theoretical mean and variance?
    4. Explain why changing n and changing size represent different changes to the experiment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    1. With n = 10: p = 0.2 gives mean 2 and right skew; p = 0.5 gives mean 5 and symmetry; p = 0.8 gives mean 8 and left skew.
    2. No. Increasing the number of repeated experiments changes the simulated data, not the theoretical distribution.
    3. The mean increases from 5 to 10; the variance increases from 2.5 to 5.
    4. n counts trials within each experiment; size counts repeated experiments. Changing n changes the model, while changing size changes the amount of simulated data.

    **Discussion:** Increasing size changes how much data we simulate, not the model.
    Increasing n changes the number of trials within each experiment and therefore the distribution of \(X\).
        """)})
    return


@app.cell(hide_code=True)
def _(binom, mo):
    _example_text = rf"""
    ## 5. Separate worked example with fixed parameters
    The sliders above do not change this example: here n = 100 and p = 0.05.

    **Example:** It is known that 5% of adults who take a certain medication experience negative side effects. 
    We have a random sample of 100 patients, and we want to calculate the probability that:

    - a) Exactly 5 patients experience side effects.
    - b) 5 patients or fewer experience side effects.
    - c) More than 5 patients experience side effects.
    - d) Between 1 and 10 patients, inclusive, experience side effects.

    Assume patients’ outcomes are independent and each patient has the same probability of experiencing 
    side effects.

    Let \(X\) be the number of patients experiencing side effects. We use a binomial distribution with 
    \(n=100\) and \(p=0.05\).

    | Question | Mathematical event | SciPy calculation | Result |
    |---|---|---|--- |
    | Exactly 5 | \(X\) = 5 | `binom.pmf(5, n=100, p=0.05)` | {binom.pmf(5, n=100, p=0.05):.4f} |
    | At most 5 | \(X\) ≤ 5 | `binom.cdf(5, n=100, p=0.05)` | {binom.cdf(5, n=100, p=0.05):.4f} |
    | More than 5 | \(X\) > 5 | `binom.sf(5, n=100, p=0.05)` | {binom.sf(5, n=100, p=0.05):.4f} |
    | Between 1 and 10, inclusive | 1 ≤ \(X\) ≤ 10 | `binom.cdf(10, n=100, p=0.05) - binom.cdf(0, n=100, p=0.05)` | {binom.cdf(10, n=100, p=0.05) - binom.cdf(0, n=100, p=0.05):.4f} |

    The **probability mass function**, **pmf**, gives the probability of an exact count. 

    The **cumulative distribution function**, **cdf**, gives the probability of that count or fewer.

    For **more than 5**, use the survival function, **sf(5)**. This equals 1 minus the probability of **5 or fewer**. 

    For **1 through 10**, subtract the probability of zero from the probability of **10 or fewer**.
    """
    mo.md(_example_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Try it yourself

    In the same example, 5% of adults taking a certain medication experience
    negative side effects. We consider 100 independent patients, each with
    the same probability of side effects.

    1. What is the probability that no patients experience side effects?
    2. What is the probability that at least one patient experiences side effects?
    3. What is the probability that at least five patients experience side effects?

    **Write your script in the next cell:**
    - Replace **None** with your calculations.
    - Use **binom.pmf()**, **binom.cdf()**, or **binom.sf()**, with **n=100** and **p=0.05**.
    - Run the cell to display your results, then expand **Show answers** to compare.

    **Remember:** “at least five” includes five.
    """)
    return


@app.cell
def _():
    # Replace None with your calculations.
    answer_1 = None
    answer_2 = None
    answer_3 = None

    {"No patients": answer_1, "At least one patient": answer_2, "At least five patients": answer_3}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show answers": mo.md("""
    1. **No patients:** P(X = 0) ≈ 0.0059. Use the PMF for exactly zero.
    2. **At least one patient:** P(X ≥ 1) ≈ 0.9941. This excludes zero: SF(0) = 1 − CDF(0).
    3. **At least five patients:** P(X ≥ 5) ≈ 0.5640. This excludes zero through four, so use SF(4).

    ```python
    answer_1 = binom.pmf(0, n=100, p=0.05)
    answer_2 = binom.sf(0, n=100, p=0.05)
    answer_3 = binom.sf(4, n=100, p=0.05)
    ```
    """)
    })
    return


@app.cell(hide_code=True)
def _(binom, mo):
    _exact = binom.pmf(8, n=10, p=0.8)
    _at_least = binom.sf(7, n=10, p=0.8)
    mo.md(f"""
    ## Return to the opening challenge
    For ten independent attempts with p = 0.8:

    - Exactly eight successes: **{_exact:.4f}**.
    - Eight or more successes: **{_at_least:.4f}**.

    Eight or more includes eight, nine, and ten. These outcomes are mutually exclusive,
    so their probabilities add. Exactly eight is only one part of that event.
    The expected count is eight; an expected count is not a guaranteed outcome.

    ### Try it yourself
    For n = 10 and p = 0.8, calculate the probability of nine or ten successes.
    Explain why it equals the difference between the two probabilities above.
    """)
    return


@app.cell(hide_code=True)
def _(binom, mo):
    mo.accordion({"Show answers": mo.md(f"P(X ≥ 9) = P(X ≥ 8) − P(X = 8) = **{binom.sf(8, n=10, p=0.8):.4f}**. Use `binom.sf(8, n=10, p=0.8)`. Subtracting exactly eight leaves nine and ten.")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Descriptive variance versus estimation
    The observed variance in the tables uses `ddof=0`: it describes the simulated observations.
    An unbiased estimator of the population variance uses `ddof=1` for independent,
    identically distributed observations. Neither makes the observed variance exactly
    match the theoretical variance.

    ## Conclusions
    - One Bernoulli observation is 0 or 1. A binomial observation counts successes in n independent trials with the same p.
    - Changing n or p changes the model. Changing the number of experiments changes the amount of simulated data.
    - The binomial mean is np and its variance is np(1−p).
    - Observed frequencies fluctuate. Agreement with theory need not improve at every sample-size increase.
    - Use PMF for exactly k, CDF for at most k, and SF(k−1) for at least k.

    ## Check your understanding
    1. For ten attempts, how does one Bernoulli outcome differ from one binomial observation?
    2. For n = 20 and p = 0.3, what are the theoretical mean and variance?
    3. Does doubling the number of simulated experiments change the theoretical mean?
    4. Why does “at least eight” use `sf(7)` rather than `sf(8)`?
    5. Would a binomial model fit attempts whose success probability changes after each failure?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md("""
    1. A Bernoulli outcome records success or failure for one attempt; a binomial observation counts successes across ten attempts.
    2. Mean = 6; variance = 4.2.
    3. No. It changes the simulated data, not n or p.
    4. SF(7) includes eight and all larger counts; SF(8) excludes eight.
    5. Not under the usual binomial assumptions: p must remain the same across trials.
    """)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reference
    Unpingco, J. (2019). *Python for Probability, Statistics, and Machine Learning*. Springer, Chapter 2.
    """)
    return


if __name__ == "__main__":
    app.run()
