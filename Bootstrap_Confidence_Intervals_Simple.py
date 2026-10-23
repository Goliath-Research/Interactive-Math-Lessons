# /// script
# dependencies = ["marimo", "numpy", "pandas", "matplotlib"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Bootstrap Confidence Intervals Step by Step

    ## Learning goals
    - Explain sampling with replacement.
    - Distinguish observations, resamples, and bootstrap means.
    - Build a percentile confidence interval for a population mean.
    - Explain what more resamples can and cannot improve.

    ## 1. Start with a question
    An online store wants to estimate its average delivery time. We have one small,
    **illustrative sample of 12 independent orders**, measured in days:

    **2, 3, 3, 4, 4, 4, 5, 5, 6, 7, 8, 9**

    **Predict first:** Would another sample of 12 orders necessarily have the same mean?
    How could we explore uncertainty without collecting thousands of new samples?

    Assume these orders reasonably represent the deliveries of interest. This tiny
    sample makes the mechanics visible; it does not guarantee accurate interval coverage.
    """)
    return


@app.cell
def _(np):
    delivery_times = np.array([2, 3, 3, 4, 4, 4, 5, 5, 6, 7, 8, 9], dtype=float)
    # Each row contains one resample of 12 source-order indices.
    bootstrap_indices = np.random.default_rng(2026).integers(0, len(delivery_times), size=(4000, len(delivery_times)))
    bootstrap_means = delivery_times[bootstrap_indices].mean(axis=1)
    return bootstrap_indices, bootstrap_means, delivery_times


@app.cell
def _(np, pd, plt):
    def percentile_interval(replicates, confidence=95):
        """Return equal-tail percentiles of bootstrap statistics."""
        tail = (100 - confidence) / 2
        return np.percentile(replicates, [tail, 100 - tail])


    def sample_view(sample, indices, resample_number):
        """Show the source order and delivery time for each draw."""
        selected = indices[resample_number - 1]
        table = pd.DataFrame({"Draw": np.arange(1, len(sample) + 1),
                              "Original order": selected + 1,
                              "Days": sample[selected]})
        fig, ax = plt.subplots(figsize=(9, 4))
        positions = np.arange(1, len(sample) + 1)
        ax.scatter(positions, sample, s=85, color="#2563eb", label="Original orders")
        counts = np.bincount(selected, minlength=len(sample))
        for order, days, count in zip(positions, sample, counts):
            ax.annotate(f"{count} draw" + ("s" if count != 1 else ""), (order, days), xytext=(0, 12),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set(xlabel="Original order", ylabel="Delivery time (days)",
               xticks=positions, ylim=(0, 11), title=f"Resample {resample_number}: how often each order was selected")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        plt.close(fig)
        return fig, table, float(sample[selected].mean())


    def interval_view(sample, means, resamples=4000, confidence=95, selected_number=1):
        """Compare observations with bootstrap means and mark a percentile interval."""
        shown = means[:resamples]
        low, high = percentile_interval(shown, confidence)
        fig, axes = plt.subplots(2, 1, figsize=(9, 7))
        axes[0].hist(sample, bins=np.arange(1.5, 10.5, 1), color="#94a3b8", edgecolor="white")
        axes[0].axvline(sample.mean(), color="#2563eb", label="Original sample mean")
        axes[0].set(title="Original observations: 12 delivery times", xlabel="Delivery time (days)", ylabel="Orders")
        axes[1].hist(shown, bins=np.linspace(2, 9, 40), color="#93c5fd", edgecolor="white")
        axes[1].axvline(low, color="#dc2626", label=f"{confidence}% interval endpoints")
        axes[1].axvline(high, color="#dc2626")
        axes[1].axvline(means[selected_number - 1], color="#15803d", linestyle="--", label="Inspected resample mean")
        axes[1].set(title=f"{resamples:,} bootstrap means", xlabel="Mean delivery time (days)", ylabel="Resamples", xlim=(2, 9))
        for ax in axes:
            ax.set_title(ax.get_title(), pad=38)
            ax.set_facecolor("white")
            ax.grid(axis="y", alpha=0.2)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
        fig.tight_layout(h_pad=4)
        plt.close(fig)
        summary = pd.DataFrame({"Quantity": ["Observed orders", "Original mean (days)", "Bootstrap resamples", "Confidence (%)", "Lower endpoint (days)", "Upper endpoint (days)", "Interval width (days)"],
                                "Value": [len(sample), sample.mean(), resamples, confidence, low, high, high-low]})
        return fig, summary

    return interval_view, sample_view


@app.cell
def _(delivery_times, mo):
    mo.md(f"**Observed sample mean: {delivery_times.mean():.4f} days.**")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Build one bootstrap sample
    Draw 12 observations **with replacement** from the 12 observed orders.
    After selecting an order, leave it available for the next draw.

    A bootstrap sample has the **same size** as the original. It can repeat some orders
    and omit others, but it cannot introduce an unobserved delivery time.

    Move the resample number to inspect different resamples. Match each selected order
    in the table to the original observations. Different orders can share the same time.

    ### Try it yourself
    1. Which orders appear more than once in the displayed resample?
    2. How many draws are there? Must all 12 original orders appear?
    3. Could a resample include a delivery time of 10 days?
    """)
    return


@app.cell
def _(mo):
    resample_number = mo.ui.slider(start=1, stop=20, step=1, value=1, show_value=True, label="Inspect resample")
    resample_number
    return (resample_number,)


@app.cell
def _(bootstrap_indices, delivery_times, mo, resample_number, sample_view):
    _fig, _table, _mean = sample_view(delivery_times, bootstrap_indices, resample_number.value)
    mo.vstack([_fig, mo.Html(_table.round(4).to_html(index=False, border=0)), mo.md(f"**This resample mean: {_mean:.4f} days.**")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md('1. Read the “draws” annotations: an order is repeated whenever its count exceeds one.\n2. There are 12 draws. Some original orders may be omitted.\n3. No. Ten days is not among the observed times. A real future delivery could still take ten days.')})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. From one mean to many means
    Repeat the resampling process and calculate **one mean per resample**.
    These means are called **bootstrap replicates**. Their distribution approximates
    how the sample mean varies across new original samples, under suitable conditions.

    The upper plot shows **individual delivery times**. The lower plot shows
    **means of bootstrap samples**. Their horizontal axes have the same units,
    but the values represent different things.

    Start with 100 resamples, then increase to 1,000 and 4,000. Each setting uses a prefix
    of the same reproducible collection of resamples. No new observed orders are added.

    ## 4. Read a percentile interval
    A **95% percentile bootstrap confidence interval** uses the 2.5th and 97.5th
    percentiles of the **bootstrap means**. The middle 95% of those means lies between
    the endpoints, apart from finite-sample quantile details.

    **Predict first:** For the same bootstrap means, will a 99% interval be wider or narrower
    than a 90% interval? Change the confidence level to check.
    """)
    return


@app.cell
def _(mo):
    resample_count = mo.ui.slider(steps=[100, 500, 1000, 4000], value=100, show_value=True, label="Bootstrap resamples")
    confidence_level = mo.ui.slider(steps=[90, 95, 99], value=95, show_value=True, label="Confidence (%)")
    mo.vstack([resample_count, confidence_level])
    return confidence_level, resample_count


@app.cell
def _(
    bootstrap_means,
    confidence_level,
    delivery_times,
    interval_view,
    mo,
    resample_count,
    resample_number,
):
    _fig, _summary = interval_view(delivery_times, bootstrap_means, resample_count.value, confidence_level.value, resample_number.value)
    mo.vstack([_fig, mo.Html(_summary.round(4).to_html(index=False, border=0, float_format='%.4f'))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. What does the interval mean?
    The interval estimates the **population mean delivery time**, in days.
    It is **not** a range intended to contain 95% of individual deliveries.

    A 95% confidence level describes a procedure: across repeated original samples,
    a well-calibrated procedure would include the fixed population mean about 95% of
    the time. The percentile bootstrap is approximate; actual coverage can differ,
    especially with very small or strongly skewed samples.

    Increasing the number of resamples reduces simulation noise in the endpoints.
    It does not increase the original sample size, guarantee that an interval gets
    narrower, or correct selection bias. More independent, representative observations
    usually provide more information about the population.

    ### Try it yourself
    1. Hold the resample count fixed. Compare 90%, 95%, and 99% intervals.
    2. Hold confidence at 95%. Compare 100 and 4,000 resamples. Must interval width decrease?
    3. If the store recorded only nearby deliveries, could 100,000 resamples fix the bias?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md('1. With the same replicates, higher confidence produces a wider or equal interval because it includes more of their distribution.\n2. No. More resamples make the endpoints more stable; the width can move in either direction.\n3. No. Resampling nearby deliveries cannot supply information about distant deliveries.')})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Calculate the interval yourself
    ### Try it yourself
    Use all **4,000** bootstrap means in `bootstrap_means`.
    Replace `None` to calculate a 90% percentile confidence interval, then report its width.
    Use `np.percentile()` and remember that a 90% interval leaves 5% in each tail.
    """)
    return


@app.cell
def _():
    student_interval = None
    student_width = None
    {"90% interval": student_interval, "Width in days": student_width}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md('```python\nstudent_interval = np.percentile(bootstrap_means, [5, 95])\nstudent_width = float(student_interval[1] - student_interval[0])\n```\nThe endpoints estimate the population mean delivery time; their difference is measured in days.')})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Conclusions
    - Resample the observed orders with replacement, keeping the original sample size.
    - Calculate a mean for every resample; the resulting means form a bootstrap distribution.
    - Take percentiles of the **means**, not of the individual delivery times.
    - A confidence interval for the mean describes uncertainty about a population average.
    - More resamples stabilize the calculation; they cannot repair an unrepresentative sample.

    ## Check your understanding
    1. A sample has 40 orders. How many draws are in one bootstrap sample?
    2. Is one bar in the bootstrap histogram counting individual deliveries or bootstrap means?
    3. Which percentiles give a 99% percentile interval?
    4. Does 95% confidence assign a 95% probability to the fixed population mean after this interval is calculated?
    5. What is the difference between collecting 100 more orders and generating 100 more resamples?

    ## Reference
    Adapted in scope from the Goliath Research Academy bootstrap lesson.
    Dekking et al. (2005), *A Modern Introduction to Probability and Statistics*, Chapter 18.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md('1. Forty draws, with replacement.\n2. Bootstrap means: one mean per resample.\n3. The 0.5th and 99.5th percentiles of the bootstrap means.\n4. No. In this frequentist interpretation the population mean is fixed; the interval-generating procedure varies across original samples.\n5. New orders add population information. New resamples only reuse the observations already available.')})
    return


if __name__ == "__main__":
    app.run()
