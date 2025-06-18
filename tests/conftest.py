from test_benchmark import TestModelZoo


def pytest_sessionfinish(session, exitstatus):
    if any("test_benchmark.py" in str(item.nodeid) for item in session.items):
        import matplotlib.pyplot as plt
        import numpy as np

        model_size = {}
        op_number = {}
        for name, summary in TestModelZoo.results.items():
            model_size[name] = {k: v.model_size for k, v in summary.items() if v is not None}
            op_number[name] = {
                k: sum(count for op, count in v.op_type_counts.items() if op != "Constant")
                for k, v in summary.items()
                if v is not None
            }

        # optimization_tools = ['float', 'onnxslim', 'onnxsim', 'polygraphy', 'onnxruntime']
        optimization_tools = ["float", "onnxslim", "onnxsim", "polygraphy"]
        colors = plt.get_cmap("tab10", len(optimization_tools))
        model_color_map = {key: colors(i) for i, key in enumerate(optimization_tools)}

        model_tested = sorted(list(TestModelZoo.results.keys()))

        ANALYSIS_METRICS = ["op_number", "model_size"]
        num_metrics = len(ANALYSIS_METRICS)
        fig, axes = plt.subplots(1, num_metrics, figsize=(7.5 * num_metrics, 9.5), sharey=True)
        if num_metrics == 1:
            axes = [axes]

        fig.suptitle("Benchmark Performance\n(Note: The larger the better)", fontsize=14, y=0.995)

        # Iterate through each source model's analysis results and plot on its respective subplot
        for ax_idx, src_model_key_iter in enumerate(ANALYSIS_METRICS):
            ax = axes[ax_idx]
            analysis_data = model_size if src_model_key_iter == "model_size" else op_number

            bar_width = 0.13
            x_category_centers = np.arange(len(model_tested))

            for tool_idx, lang_cat_key in enumerate(optimization_tools):
                y_values = []
                for model_display_name in model_tested:
                    baseline = analysis_data[model_display_name].get("float", 1)
                    value = analysis_data[model_display_name].get(lang_cat_key, 0)
                    y_values.append(baseline / value if value else 0)

                offset = (tool_idx - (len(optimization_tools) - 1) / 2) * bar_width
                bars = ax.bar(
                    x_category_centers + offset,
                    y_values,
                    bar_width,
                    label=lang_cat_key,
                    color=model_color_map[lang_cat_key],
                    hatch="////" if lang_cat_key == "float" else None,
                    edgecolor="black",
                )

                for bar in bars:
                    yval = bar.get_height()
                    if yval > 0.01:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            yval + 0.01 * ax.get_ylim()[1],
                            f"{yval:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )

            ax.legend(title="Optimization Tool", loc="upper left", fontsize=9)
            if ax_idx == 0:
                ax.set_ylabel("Compression Ratio (x)", fontsize=12)

            ax.set_title(
                f"{src_model_key_iter.replace('_', ' ').title()} Compression Ratio by Optimization Tool", fontsize=14
            )
            ax.set_xticks(x_category_centers)
            ax.set_xticklabels(model_tested, fontsize=10, rotation=30, ha="right")
            ax.tick_params(axis="y", labelsize=10)
            ax.set_ylim(bottom=0)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("./benchmark.png", dpi=300, bbox_inches="tight")
