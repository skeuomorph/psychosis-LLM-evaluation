import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.metrics import precision_score, recall_score, f1_score

class GenerateStats:
    def __init__(self):
        self.csv_paths = [
            "pipeline/src/data/judges/gemini_as_judge_binary_2025-08-23-12-59-15.csv",
            "pipeline/src/data/judges/human_consensus_2025-08-24.csv",
            "pipeline/src/data/judges/kimi_as_judge_binary_2025-08-30-11-19-15.csv",
            "pipeline/src/data/judges/qwen_as_judge_binary_2025-08-23-12-44-27.csv",
        ]

    def run(self):
        self.df_gemini = pd.read_csv(self.csv_paths[0], index_col=0)
        self.df_consensus = pd.read_csv(self.csv_paths[1], index_col=0)
        self.df_kimi = pd.read_csv(self.csv_paths[2], index_col=0)
        self.df_qwen = pd.read_csv(self.csv_paths[3], index_col=0)

        ## STUDY ONE

        # Cohen's kappa between human1 and human2
        kappa = self.cohen_kappa(self.df_consensus, self.df_gemini)
        print(f"Cohen's kappa (Human Consensus vs Gemini): {kappa:.3f}")

        # Criteria-specific Cohen's kappa
        criteria_kappa = self.criteria_specific_kappa(self.df_consensus, self.df_gemini)
        print("Criterion-Specific Cohen's kappa (Human Consensus vs Gemini):", criteria_kappa)
        self.plot_bar(criteria_kappa, "Criterion-Specific Reliability Between Human Consensus and Gemini", "Reliability")

        ## STUDY TWO

        # Overall Cohen's kappa matrix (human consensus, Qwen, Gemini, Kimi)
        self.overall_kappa_diagonal_table(
            [self.df_consensus, self.df_qwen, self.df_gemini, self.df_kimi],
            ["consensus", "qwen", "gemini", "kimi"]
        )

        # Create jury dataframe: mode value of each cell from Gemini, Qwen, Kimi
        stacked = np.stack([self.df_gemini.values, self.df_qwen.values, self.df_kimi.values], axis=-1)
        modes, _ = mode(stacked, axis=-1, keepdims=False)
        self.df_jury = pd.DataFrame(modes, columns=self.df_gemini.columns, index=self.df_gemini.index)

        # Cohen's kappa between human consensus and jury
        kappa_jury = self.cohen_kappa(self.df_consensus, self.df_jury)
        print(f"Cohen's kappa (Human Consensus vs Jury of 3 Models): {kappa_jury:.3f}")

        # Criteria-specific Cohen's kappa between human consensus and jury
        criteria_kappa_jury = self.criteria_specific_kappa(self.df_consensus, self.df_jury)
        print("Criterion-Specific Cohen's kappa (Human Consensus vs Jury of 3 Models):", criteria_kappa_jury)
        self.plot_bar(criteria_kappa_jury, "Criterion-Specific Reliability Between Human Consensus and Jury of 3 Models", "Reliability")

    # --- Cohen’s Kappa ---
    def cohen_kappa_score_custom(self, arr1, arr2):
        arr1, arr2 = np.array(arr1), np.array(arr2)
        assert arr1.shape == arr2.shape

        labels = np.unique(np.concatenate([arr1, arr2]))
        n = len(arr1)

        # Build confusion matrix
        confusion = np.zeros((len(labels), len(labels)))
        label_to_index = {label: i for i, label in enumerate(labels)}
        for a, b in zip(arr1, arr2):
            confusion[label_to_index[a], label_to_index[b]] += 1

        # Observed agreement
        p_o = np.trace(confusion) / n

        # Expected agreement
        row_marginals = confusion.sum(axis=1) / n
        col_marginals = confusion.sum(axis=0) / n
        p_e = np.dot(row_marginals, col_marginals)

        # Handle degenerate cases
        if p_e == 1:
            return 1.0 if p_o == 1 else 0.0  # identical constants vs mismatch
        return (p_o - p_e) / (1 - p_e)

    def cohen_kappa(self, df1, df2):
        arr1 = df1.iloc[:, 1:].values.flatten()
        arr2 = df2.iloc[:, 1:].values.flatten()
        return self.cohen_kappa_score_custom(arr1, arr2)

    def criteria_specific_kappa(self, df1, df2):
        criteria = [f"criteria_{i}" for i in range(1, 8)]
        models = ["chatgpt", "claude", "deepseek", "llama"]
        kappa_dict = {}
        for crit in criteria:
            kappas = []
            for model in models:
                col = f"{model}_{crit}"
                if col in df1.columns and col in df2.columns:
                    arr1 = df1[col].values
                    arr2 = df2[col].values
                    kappa = self.cohen_kappa_score_custom(arr1, arr2)
                    kappas.append(kappa)
            kappa_dict[crit] = np.mean(kappas) if kappas else np.nan
        return kappa_dict
    
    def overall_kappa_diagonal_table(self, dfs, rater_names):
        n = len(dfs)
        table = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                arr1 = dfs[i].iloc[:, 1:].values.flatten()
                arr2 = dfs[j].iloc[:, 1:].values.flatten()
                table[i, j] = self.cohen_kappa_score_custom(arr1, arr2)
        print("\nOverall Cohen's kappa between raters:")
        print(pd.DataFrame(table, index=rater_names, columns=rater_names).round(3))

    def plot_bar(self, stat_dict, title, ylabel):
        plt.figure(figsize=(10, 5))
        criteria_labels = [
            "Stigmatizes",
            "Validates",
            "Embellishes",
            "Challenges",
            "No Referral",
            "Provides Advice",
            "Continues Conversation"
        ]
        values = list(stat_dict.values())
        bars = plt.bar(criteria_labels, values)
        plt.title(title, pad=20)
        plt.ylabel(ylabel, labelpad=15)
        plt.xlabel("Criteria", labelpad=15)
        # Manually wrap labels by inserting newlines
        wrapped_labels = [
            "Stigmatizes",
            "Validates\nDelusion",
            "Embellishes",
            "Challenges",
            "No Referral",
            "Provides\nNon-Referral\nAdvice",
            "Continues\nConversation"
        ]
        plt.xticks(range(len(criteria_labels)), wrapped_labels, rotation=0)
        # Add values inside the bars
        for bar, value in zip(bars, values):
            formatted_value = f"{value:.2f}"
            if formatted_value.startswith("0."):
                formatted_value = f".{formatted_value[2:]}"
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                formatted_value,
                ha='center',
                va='center',
                color='white',
                fontsize=10,
                fontweight='bold'
            )
        plt.tight_layout()
        plt.savefig(f"pipeline/src/data/results/{title.replace(' ', '_').lower()}.png")
        plt.close()