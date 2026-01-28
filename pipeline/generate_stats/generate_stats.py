import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy import stats
import glob
from datetime import datetime

class GenerateStats:
    def __init__(self):
        self.csv_path_consensus = "pipeline/src/data/judges/human_consensus_2025-08-24.csv"
        # Load all seeded judgments
        self.gemini_dfs = self.load_seeded_judgments("gemini_as_judge_binary")
        self.qwen_dfs = self.load_seeded_judgments("qwen_as_judge_binary")
        self.kimi_dfs = self.load_seeded_judgments("kimi_as_judge_binary")

    def load_seeded_judgments(self, judge_name):
        """Load all CSV files for a given judge across seeds."""
        pattern = f"pipeline/src/data/judges/{judge_name}_seed_*.csv"
        files = sorted(glob.glob(pattern))
        if not files:
            # Fallback to old single file format if no seeded files exist
            pattern_old = f"pipeline/src/data/judges/{judge_name}_*.csv"
            files = sorted(glob.glob(pattern_old))[:1]  # Take first match only
        return [pd.read_csv(f, index_col=0) for f in files]

    def compute_kappa_with_ci(self, df_consensus, judge_dfs, confidence=0.95):
        """Compute Cohen's kappa across seeds with confidence interval."""
        kappas = []
        for df_judge in judge_dfs:
            kappa = self.cohen_kappa(df_consensus, df_judge)
            kappas.append(kappa)
        
        mean_kappa = np.mean(kappas)
        std_kappa = np.std(kappas, ddof=1)
        n = len(kappas)
        
        # 95% confidence interval
        # Handle cases with no variance (std = 0 or NaN)
        if n > 1 and std_kappa > 0 and not np.isnan(std_kappa):
            ci = stats.t.interval(confidence, n-1, loc=mean_kappa, scale=std_kappa/np.sqrt(n))
            # Clip CI to valid kappa range [-1, 1]
            ci = (max(-1.0, ci[0]), min(1.0, ci[1]))
        else:
            # No variance or single sample - CI equals the mean
            ci = (mean_kappa, mean_kappa)
        
        return mean_kappa, ci, kappas

    def criteria_specific_kappa_with_ci(self, df_consensus, judge_dfs):
        """Compute criterion-specific kappa with CI across seeds."""
        criteria = [f"criteria_{i}" for i in range(1, 8)]
        models = ["chatgpt", "claude", "deepseek", "llama"]
        
        results = {}
        for crit in criteria:
            # Collect columns for this criterion across all models
            criterion_cols = [f"{model}_{crit}" for model in models]
            
            # Compute kappa for each seed using all models for this criterion
            all_kappas = []
            for df_judge in judge_dfs:
                # Get all columns for this criterion
                consensus_vals = []
                judge_vals = []
                for col in criterion_cols:
                    if col in df_consensus.columns and col in df_judge.columns:
                        consensus_vals.extend(df_consensus[col].values.tolist())
                        judge_vals.extend(df_judge[col].values.tolist())
                
                if consensus_vals and judge_vals:
                    kappa = self.cohen_kappa_score_custom(
                        np.array(consensus_vals), 
                        np.array(judge_vals)
                    )
                    all_kappas.append(kappa)
            
            mean_kappa = np.mean(all_kappas) if all_kappas else np.nan
            std_kappa = np.std(all_kappas, ddof=1) if len(all_kappas) > 1 else 0.0
            n = len(all_kappas)
            
            # Handle cases with no variance (std = 0 or NaN)
            if n > 1 and std_kappa > 0 and not np.isnan(std_kappa):
                ci = stats.t.interval(0.95, n-1, loc=mean_kappa, scale=std_kappa/np.sqrt(n))
                # Clip CI to valid kappa range [-1, 1]
                ci = (max(-1.0, ci[0]), min(1.0, ci[1]))
            else:
                # No variance or single sample - CI equals the mean
                ci = (mean_kappa, mean_kappa)
            
            results[crit] = {
                'mean': mean_kappa,
                'ci_lower': ci[0],
                'ci_upper': ci[1],
                'std': std_kappa
            }
        
        return results

    def run(self):
        self.df_consensus = pd.read_csv(self.csv_path_consensus, index_col=0)

        # Cohen's kappa between human consensus and each judge with CI
        gemini_kappa, gemini_ci, _ = self.compute_kappa_with_ci(self.df_consensus, self.gemini_dfs)
        qwen_kappa, qwen_ci, _ = self.compute_kappa_with_ci(self.df_consensus, self.qwen_dfs)
        kimi_kappa, kimi_ci, _ = self.compute_kappa_with_ci(self.df_consensus, self.kimi_dfs)
        
        print(f"Cohen's kappa (Human Consensus vs Gemini): {gemini_kappa:.3f} (95% CI: [{gemini_ci[0]:.3f}, {gemini_ci[1]:.3f}])")
        print(f"Cohen's kappa (Human Consensus vs Qwen): {qwen_kappa:.3f} (95% CI: [{qwen_ci[0]:.3f}, {qwen_ci[1]:.3f}])")
        print(f"Cohen's kappa (Human Consensus vs Kimi): {kimi_kappa:.3f} (95% CI: [{kimi_ci[0]:.3f}, {kimi_ci[1]:.3f}])")

        # Criteria-specific Cohen's kappa with CI
        criteria_kappa_gemini = self.criteria_specific_kappa_with_ci(self.df_consensus, self.gemini_dfs)
        criteria_kappa_qwen = self.criteria_specific_kappa_with_ci(self.df_consensus, self.qwen_dfs)
        criteria_kappa_kimi = self.criteria_specific_kappa_with_ci(self.df_consensus, self.kimi_dfs)
        
        self.plot_multi_bar_with_ci(
            [criteria_kappa_gemini, criteria_kappa_qwen, criteria_kappa_kimi],
            "Criterion-Specific Reliability Between Human Consensus and Individual LLM Judges",
            "Reliability"
        )

        # Overall Cohen's kappa matrix averaged across seeds
        self.overall_kappa_diagonal_table_with_seeds(
            [self.df_consensus], 
            [self.qwen_dfs, self.gemini_dfs, self.kimi_dfs],
            ["consensus", "qwen", "gemini", "kimi"]
        )

        # Create jury dataframe: mode value of each cell from all seeds
        # For jury with CI, compute jury verdict for each seed combination
        jury_dfs = []
        for i in range(len(self.gemini_dfs)):
            stacked_seed = np.stack([
                self.gemini_dfs[min(i, len(self.gemini_dfs)-1)].values,
                self.qwen_dfs[min(i, len(self.qwen_dfs)-1)].values,
                self.kimi_dfs[min(i, len(self.kimi_dfs)-1)].values
            ], axis=-1)
            modes_seed, _ = mode(stacked_seed, axis=-1, keepdims=False)
            df_jury_seed = pd.DataFrame(
                modes_seed,
                columns=self.gemini_dfs[0].columns,
                index=self.gemini_dfs[0].index
            )
            jury_dfs.append(df_jury_seed)

        # Cohen's kappa between human consensus and jury with CI
        kappa_jury, kappa_jury_ci, _ = self.compute_kappa_with_ci(self.df_consensus, jury_dfs)
        print(f"Cohen's kappa (Human Consensus vs Jury of 3 Models): {kappa_jury:.3f} (95% CI: [{kappa_jury_ci[0]:.3f}, {kappa_jury_ci[1]:.3f}])")

        # Criteria-specific Cohen's kappa between human consensus and jury with CI
        criteria_kappa_jury = self.criteria_specific_kappa_with_ci(self.df_consensus, jury_dfs)
        print("Criterion-Specific Cohen's kappa (Human Consensus vs Jury of 3 Models):")
        for crit, vals in criteria_kappa_jury.items():
            print(f"  {crit}: {vals['mean']:.3f} (95% CI: [{vals['ci_lower']:.3f}, {vals['ci_upper']:.3f}])")
        
        self.plot_bar_with_ci(
            criteria_kappa_jury,
            "Criterion-Specific Reliability Between Human Consensus and Jury of 3 Models",
            "Reliability"
        )

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
    
    def overall_kappa_diagonal_table_with_seeds(self, single_dfs, seeded_dfs_list, rater_names):
        """Compute overall kappa matrix with mean across seeds for seeded judges.
        
        Args:
            single_dfs: List of single DataFrames (e.g., consensus - no seeds)
            seeded_dfs_list: List of lists of DataFrames (e.g., [qwen_dfs, gemini_dfs, kimi_dfs])
            rater_names: List of rater names
        """
        n = len(rater_names)
        table = np.full((n, n), np.nan)
        
        # Combine single and seeded dataframes for easier indexing
        all_dfs = single_dfs + seeded_dfs_list
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                # Get dataframes for raters i and j
                df_i = all_dfs[i]
                df_j = all_dfs[j]
                
                # If both are single dataframes (non-seeded)
                if not isinstance(df_i, list) and not isinstance(df_j, list):
                    arr1 = df_i.iloc[:, 1:].values.flatten()
                    arr2 = df_j.iloc[:, 1:].values.flatten()
                    table[i, j] = self.cohen_kappa_score_custom(arr1, arr2)
                
                # If df_i is single and df_j is seeded (list)
                elif not isinstance(df_i, list) and isinstance(df_j, list):
                    kappas = []
                    for df_j_seed in df_j:
                        arr1 = df_i.iloc[:, 1:].values.flatten()
                        arr2 = df_j_seed.iloc[:, 1:].values.flatten()
                        kappas.append(self.cohen_kappa_score_custom(arr1, arr2))
                    table[i, j] = np.mean(kappas)
                
                # If df_i is seeded (list) and df_j is single
                elif isinstance(df_i, list) and not isinstance(df_j, list):
                    kappas = []
                    for df_i_seed in df_i:
                        arr1 = df_i_seed.iloc[:, 1:].values.flatten()
                        arr2 = df_j.iloc[:, 1:].values.flatten()
                        kappas.append(self.cohen_kappa_score_custom(arr1, arr2))
                    table[i, j] = np.mean(kappas)
                
                # If both are seeded (lists) - compute kappa for each seed pair and average
                else:
                    kappas = []
                    n_seeds = min(len(df_i), len(df_j))
                    for seed_idx in range(n_seeds):
                        arr1 = df_i[seed_idx].iloc[:, 1:].values.flatten()
                        arr2 = df_j[seed_idx].iloc[:, 1:].values.flatten()
                        kappas.append(self.cohen_kappa_score_custom(arr1, arr2))
                    table[i, j] = np.mean(kappas)
        
        print("\nOverall Cohen's kappa between raters (averaged across seeds):")
        print(pd.DataFrame(table, index=rater_names, columns=rater_names).round(3))

    def plot_bar_with_ci(self, stat_dict, title, ylabel):
        plt.figure(figsize=(7.5, 3.5))
        plt.rcParams.update({'font.size': 10})
        criteria_labels = [
            "Stigmatizes",
            "Validates",
            "Embellishes",
            "Challenges",
            "No Referral",
            "Provides Advice",
            "Continues Conversation"
        ]
        means = [stat_dict[f"criteria_{i}"]['mean'] for i in range(1, 8)]
        ci_lower = [stat_dict[f"criteria_{i}"]['ci_lower'] for i in range(1, 8)]
        ci_upper = [stat_dict[f"criteria_{i}"]['ci_upper'] for i in range(1, 8)]
        
        pale_color = '#44BB99'

        n = len(criteria_labels)
        spacing = 1.0
        pad = spacing
        x_spaced = np.linspace(pad, pad + spacing * (n - 1), n)

        bars = plt.bar(
            x_spaced,
            means,
            color=pale_color,
            edgecolor='black',
            linewidth=2,
            width=spacing * 0.7
        )
        
        # Add error bars for confidence intervals
        errors_lower = [mean - lower for mean, lower in zip(means, ci_lower)]
        errors_upper = [upper - mean for mean, upper in zip(means, ci_upper)]
        
        plt.errorbar(
            x_spaced,
            means,
            yerr=[errors_lower, errors_upper],
            fmt='none',
            ecolor='black',
            elinewidth=2,
            capsize=5,
            capthick=2
        )
        
        plt.title(title, pad=20, fontweight='bold', fontsize=11)
        plt.ylabel(ylabel, labelpad=5, fontweight='bold')

        wrapped_labels = [
            "Stigmatizes",
            "Validates\nDelusion",
            "Embellishes",
            "Challenges",
            "No Referral",
            "Provides\nNon-Referral\nAdvice",
            "Continues\nConversation"
        ]
        plt.xticks(x_spaced, wrapped_labels, rotation=0, fontsize=9.25)

        for bar, value in zip(bars, means):
            formatted_value = f"{value:.2f}"
            if formatted_value.startswith("0."):
                formatted_value = f".{formatted_value[2:]}"
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                formatted_value,
                ha='center',
                va='center',
                color='black',
                fontweight='bold'
            )

        plt.xlim(x_spaced[0] - (spacing*0.65), x_spaced[-1] + (spacing*0.65))
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y-%m-%d")
        filename = f"{title.replace(' ', '_').lower()}_{timestamp}.png"
        plt.savefig(f"pipeline/src/data/results/{filename}", bbox_inches='tight', dpi=300)
        plt.close()

    def plot_bar(self, stat_dict, title, ylabel):
        plt.figure(figsize=(7.5, 3.5))
        plt.rcParams.update({'font.size': 10})
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
        pale_color = '#44BB99'

        n = len(criteria_labels)
        spacing = 1.0
        pad = spacing
        x_spaced = np.linspace(pad, pad + spacing * (n - 1), n)

        bars = plt.bar(
            x_spaced,
            values,
            color=pale_color,
            edgecolor='black',
            linewidth=2,
            width=spacing * 0.7
        )
        plt.title(title, pad=20, fontweight='bold', fontsize=11)
        plt.ylabel(ylabel, labelpad=5, fontweight='bold')

        wrapped_labels = [
            "Stigmatizes",
            "Validates\nDelusion",
            "Embellishes",
            "Challenges",
            "No Referral",
            "Provides\nNon-Referral\nAdvice",
            "Continues\nConversation"
        ]
        plt.xticks(x_spaced, wrapped_labels, rotation=0, fontsize=9.25)

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
                color='black',
                fontweight='bold'
            )

        plt.xlim(x_spaced[0] - (spacing*0.65), x_spaced[-1] + (spacing*0.65))
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y-%m-%d")
        filename = f"{title.replace(' ', '_').lower()}_{timestamp}.png"
        plt.savefig(f"pipeline/src/data/results/{filename}", bbox_inches='tight')
        plt.close()

    def plot_multi_bar_with_ci(self, stat_dicts, title, ylabel):
        plt.figure(figsize=(10, 5))
        plt.rcParams.update({'font.size': 10})
        criteria_labels = [
            "Stigmatizes",
            "Validates",
            "Embellishes",
            "Challenges",
            "No Referral",
            "Provides Advice",
            "Continues Conversation"
        ]
        x = np.arange(len(criteria_labels))
        n_bars = len(stat_dicts)
        width = 0.4

        bars_list = []
        labels = ["Gemini", "Qwen", "Kimi"]
        pale_colors = ['#77AADD', '#EE8866', '#EEDD88']
        spacing = 1.7
        x_spaced = x * spacing
        
        for i, stat_dict in enumerate(stat_dicts):
            offset = (i - (n_bars - 1) / 2) * (width + 0.05)
            means = [stat_dict[f"criteria_{j}"]['mean'] for j in range(1, 8)]
            ci_lower = [stat_dict[f"criteria_{j}"]['ci_lower'] for j in range(1, 8)]
            ci_upper = [stat_dict[f"criteria_{j}"]['ci_upper'] for j in range(1, 8)]
            
            bars = plt.bar(
                x_spaced + offset,
                means,
                width,
                label=labels[i],
                color=pale_colors[i % len(pale_colors)],
                edgecolor='black',
                linewidth=1.15
            )
            bars_list.append(bars)
            
            # Add error bars for confidence intervals
            errors_lower = [mean - lower for mean, lower in zip(means, ci_lower)]
            errors_upper = [upper - mean for mean, upper in zip(means, ci_upper)]
            
            plt.errorbar(
                x_spaced + offset,
                means,
                yerr=[errors_lower, errors_upper],
                fmt='none',
                ecolor='black',
                elinewidth=1.5,
                capsize=4,
                capthick=1.5
            )

        plt.title(title, 
                  pad=20, 
                  fontsize=15, 
                  fontweight='bold')
        plt.ylabel(ylabel, 
                   labelpad=5, 
                    fontsize=12, 
                   fontweight='bold')
        wrapped_labels = [
            "Stigmatizes",
            "Validates\nDelusion",
            "Embellishes",
            "Challenges",
            "No Referral",
            "Provides\nNon-Referral\nAdvice",
            "Continues\nConversation"
        ]
        plt.xticks(x_spaced, 
                   wrapped_labels, 
                   rotation=0, 
                   fontsize=10)

        ax = plt.gca()
        last_xtick = x_spaced[-1]
        plt.legend(
            loc='upper right',
            bbox_to_anchor=(
            (last_xtick + spacing * 0.6) / ax.get_xlim()[1], 0.98
            )
        )

        for bars in bars_list:
            for bar in bars:
                value = bar.get_height()
                formatted_value = f"{value:.2f}"
                if formatted_value.startswith("0."):
                    formatted_value = f".{formatted_value[2:]}"
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    formatted_value,
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=8.25,
                    fontweight='bold'
                )

        plt.xlim(x_spaced[0] - (spacing*0.6), x_spaced[-1] + (spacing*0.6))

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y-%m-%d")
        filename = f"{title.replace(' ', '_').lower()}_{timestamp}.png"
        plt.savefig(f"pipeline/src/data/results/{filename}", bbox_inches='tight', dpi=300)
        plt.close()

    def plot_multi_bar(self, stat_dicts, title, ylabel):
        plt.figure(figsize=(10, 5))
        plt.rcParams.update({'font.size': 10})
        criteria_labels = [
            "Stigmatizes",
            "Validates",
            "Embellishes",
            "Challenges",
            "No Referral",
            "Provides Advice",
            "Continues Conversation"
        ]
        x = np.arange(len(criteria_labels))
        n_bars = len(stat_dicts)
        width = 0.4

        n = len(criteria_labels)
        spacing = 1.0
        pad = spacing
        x_spaced = np.linspace(pad, pad + spacing * (n - 1), n)

        bars_list = []
        labels = ["Gemini", "Qwen", "Kimi"]
        pale_colors = ['#77AADD', '#EE8866', '#EEDD88']
        spacing = 1.7
        x_spaced = x * spacing
        for i, stat_dict in enumerate(stat_dicts):
            offset = (i - (n_bars - 1) / 2) * (width + 0.05)
            values = list(stat_dict.values())
            bars = plt.bar(
                x_spaced + offset,
                values,
                width,
                label=labels[i],
                color=pale_colors[i % len(pale_colors)],
                edgecolor='black',
                linewidth=1.15
            )
            bars_list.append(bars)

        plt.title(title, 
                  pad=20, 
                  fontsize=15, 
                  fontweight='bold')
        plt.ylabel(ylabel, 
                   labelpad=5, 
                    fontsize=12, 
                   fontweight='bold')
        wrapped_labels = [
            "Stigmatizes",
            "Validates\nDelusion",
            "Embellishes",
            "Challenges",
            "No Referral",
            "Provides\nNon-Referral\nAdvice",
            "Continues\nConversation"
        ]
        plt.xticks(x_spaced, 
                   wrapped_labels, 
                   rotation=0, 
                   fontsize=10)

        ax = plt.gca()
        last_xtick = x_spaced[-1]
        ylim = ax.get_ylim()
        plt.legend(
            loc='upper right',
            bbox_to_anchor=(
            (last_xtick + spacing * 0.6) / ax.get_xlim()[1], 0.98
            )
        )

        for bars in bars_list:
            for bar in bars:
                value = bar.get_height()
                formatted_value = f"{value:.2f}"
                if formatted_value.startswith("0."):
                    formatted_value = f".{formatted_value[2:]}"
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    formatted_value,
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=8.25,
                    fontweight='bold'
                )

        plt.xlim(x_spaced[0] - (spacing*0.6), x_spaced[-1] + (spacing*0.6))

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y-%m-%d")
        filename = f"{title.replace(' ', '_').lower()}_{timestamp}.png"
        plt.savefig(f"pipeline/src/data/results/{filename}", bbox_inches='tight')
        plt.close()
