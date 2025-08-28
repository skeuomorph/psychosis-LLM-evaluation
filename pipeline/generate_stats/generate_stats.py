import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class GenerateStats:
    def __init__(self):
        self.csv_paths = [
            "pipeline/src/data/judges/gemini_as_judge_binary_2025-08-23-12-59-15.csv",
            "pipeline/src/data/judges/human2_base_scenarios_manual_validation.csv",
            "pipeline/src/data/judges/magistral_as_judge_binary_2025-08-23-13-07-35.csv",
            "pipeline/src/data/judges/human1_base_scenarios_manual_validation.csv",
            "pipeline/src/data/judges/qwen_as_judge_binary_2025-08-23-12-44-27.csv"
        ]

    def run(self):
        self.df_gemini = pd.read_csv(self.csv_paths[0])
        self.df_human2 = pd.read_csv(self.csv_paths[1])
        self.df_magistral = pd.read_csv(self.csv_paths[2])
        self.df_human1 = pd.read_csv(self.csv_paths[3])
        self.df_qwen = pd.read_csv(self.csv_paths[4])

        # Cohen's kappa between human1 and human2
        kappa = self.cohen_kappa(self.df_human1, self.df_human2)
        print(f"Cohen's kappa (human1 vs human2): {kappa:.3f}")

        # Criteria-specific stats
        criteria_kappa = self.criteria_specific_kappa(self.df_human1, self.df_human2)
        self.plot_bar(criteria_kappa, "Human Raters Criteria-specific Cohen's Kappa", "Kappa")

        # Criteria-specific Fleiss' Kappa (human2, human1, Qwen)
        fleiss_kappa_kmq = self.criteria_specific_fleiss_kappa([self.df_human2, self.df_human1, self.df_qwen])
        self.plot_bar(fleiss_kappa_kmq, "Criteria-specific Fleiss' Kappa (human2, human1, Qwen)", "Fleiss' Kappa")

        # Criteria-specific Fleiss' Kappa (human2, human1, Gemini)
        fleiss_kappa_kmg = self.criteria_specific_fleiss_kappa([self.df_human2, self.df_human1, self.df_gemini])
        self.plot_bar(fleiss_kappa_kmg, "Criteria-specific Fleiss' Kappa (human2, human1, Gemini)", "Fleiss' Kappa")

        # Fleiss' Kappa stats (overall)
        fleiss_stats = self.fleiss_kappa_stats()
        for desc, value in fleiss_stats.items():
            print(f"{desc}: {value:.3f}")

    # --- Custom Cohen’s Kappa (faithful to definition) ---
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

        print(p_o)
        print(p_e)

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

    def criteria_specific_fleiss_kappa(self, dfs):
        criteria = [f"criteria_{i}" for i in range(1, 8)]
        models = ["chatgpt", "claude", "deepseek", "llama"]
        fleiss_dict = {}
        for crit in criteria:
            cols = [f"{model}_{crit}" for model in models if f"{model}_{crit}" in dfs[0].columns]
            # Only use columns present in all dfs
            valid_cols = [col for col in cols if all(col in df.columns for df in dfs)]
            if not valid_cols:
                fleiss_dict[crit] = np.nan
                continue
            ratings = []
            for df in dfs:
                ratings.append(df[valid_cols].values.flatten())
            ratings = np.array(ratings).T  # shape: (n_items, n_raters)
            fleiss_dict[crit] = self._fleiss_kappa_from_ratings(ratings)
        return fleiss_dict

    def _fleiss_kappa_from_ratings(self, ratings):
        n_items, n_raters = ratings.shape
        categories = np.unique(ratings[~np.isnan(ratings)])
        n_categories = len(categories)
        cat_to_index = {cat: i for i, cat in enumerate(categories)}

        # Build count matrix: (n_items, n_categories)
        category_counts = np.zeros((n_items, n_categories))
        for i in range(n_items):
            for j in range(n_raters):
                val = ratings[i, j]
                if not np.isnan(val):
                    category_counts[i, cat_to_index[val]] += 1

        # Compute P_i (agreement for item i)
        if n_raters < 2:
            return np.nan
        P_i = ((category_counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
        P_bar = P_i.mean()

        # Compute expected agreement
        p = category_counts.sum(axis=0) / (n_items * n_raters)
        P_e_bar = (p ** 2).sum()

        return (P_bar - P_e_bar) / (1 - P_e_bar) if (1 - P_e_bar) != 0 else np.nan

    def plot_bar(self, stat_dict, title, ylabel):
        plt.figure(figsize=(10, 5))
        plt.bar(stat_dict.keys(), stat_dict.values())
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_').lower()}.png")
        plt.close()

    def fleiss_kappa_stats(self):
        stats = {}
        stats["Fleiss' Kappa (human1, human2, Qwen)"] = self.fleiss_kappa([self.df_human1, self.df_human2, self.df_qwen])
        stats["Fleiss' Kappa (human1, human2, Gemini)"] = self.fleiss_kappa([self.df_human1, self.df_human2, self.df_gemini])
        stats["Fleiss' Kappa (human1, human2, Magistral)"] = self.fleiss_kappa([self.df_human1, self.df_human2, self.df_magistral])
        stats["Fleiss' Kappa (Qwen, Gemini, Magistral)"] = self.fleiss_kappa([self.df_qwen, self.df_gemini, self.df_magistral])
        stats["Fleiss' Kappa (All 5)"] = self.fleiss_kappa([self.df_human1, self.df_human2, self.df_qwen, self.df_gemini, self.df_magistral])
        stats["Fleiss' Kappa (human1, human2, Qwen, Gemini)"] = self.fleiss_kappa([self.df_human1, self.df_human2, self.df_qwen, self.df_gemini])
        return stats

    # --- Fleiss’ Kappa (overall, all columns) ---
    def fleiss_kappa(self, dfs):
        cols = [col for col in dfs[0].columns if col != "id"]
        ratings = []
        for df in dfs:
            ratings.append(df[cols].values.flatten())
        ratings = np.array(ratings).T  # shape: (n_items, n_raters)

        n_items, n_raters = ratings.shape
        categories = np.unique(ratings[~np.isnan(ratings)])
        n_categories = len(categories)
        cat_to_index = {cat: i for i, cat in enumerate(categories)}

        # Build count matrix: (n_items, n_categories)
        category_counts = np.zeros((n_items, n_categories))
        for i in range(n_items):
            for j in range(n_raters):
                val = ratings[i, j]
                if not np.isnan(val):
                    category_counts[i, cat_to_index[val]] += 1

        if n_raters < 2:
            return np.nan
        # Compute P_i (agreement for item i)
        P_i = ((category_counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
        P_bar = P_i.mean()

        # Compute expected agreement
        p = category_counts.sum(axis=0) / (n_items * n_raters)
        P_e_bar = (p ** 2).sum()

        return (P_bar - P_e_bar) / (1 - P_e_bar) if (1 - P_e_bar) != 0 else np.nan