import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
import random
import argparse
from collections import defaultdict
from embeddingtest import MMDTest


class InfluenceRanker:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_random_seed()
        self._setup_directories()
        self._initializer_groups()

    def _set_random_seed(self):
        """Set random seed for reproducibility."""
        self.seed = self.args.random_state
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _setup_directories(self):
        """Setup directory paths relative to parent directory."""
        self.root_dir = Path(__file__).resolve().parents[1]
        base = self.root_dir / "adni_results"
        self.embed_dir = base / "embeddings"
        self.inf_dir = base / "infscores"
        self.param_dir = base / "params"

        os.makedirs(self.embed_dir, exist_ok=True)
        os.makedirs(self.inf_dir, exist_ok=True)
        os.makedirs(self.param_dir, exist_ok=True)

    def _load_embeddings(self):
        """
        Load embeddings of original data set.
        Returns:
            self.gr1_embed (np.ndarray): Embeddings of group 1.
            self.gr2_embed (np.ndarray): Embeddings of group 2.
        """
        self.n, self.m = self.args.n, self.args.m
        file_path = os.path.join(self.embed_dir, f"{self.seed}_{self.args.exp}_{self.args.ckp}_{self.n + self.m}")

        path_gr1 = os.path.join(file_path, "healthy_embed.npy")
        path_gr2 = os.path.join(file_path, "unhealthy_embed.npy")

        if os.path.exists(path_gr1) and os.path.exists(path_gr2):
            print("Loading embeddings from saved numpy files...")
            self.gr1_embed = np.load(path_gr1, mmap_mode="r")
            self.gr2_embed = np.load(path_gr2, mmap_mode="r")
            self.p, self.q = len(self.gr1_embed), len(self.gr2_embed)
            # compute p-value for original dataset
            self.d = self._compute_test_statistic(self.gr1_embed, self.gr2_embed)
            print(f"p_value for original data is: {self.d}")
        else:
            raise FileNotFoundError(f"One or both embedding files not found: {path_gr1}, {path_gr2}")

    def _load_infscores(self):
        """
        Load influence scores of original data set.
        Returns:
            self.gr1_scr: (np.ndarray): Influence scores of group 1.
            self.gr2_scr: (np.ndarray): Influence scores of group 2.
        """
        file_path = os.path.join(self.inf_dir, f"{self.seed}_{self.args.exp}_{self.args.ckp}_{self.n + self.m}")

        path_X = os.path.join(file_path, "healthy_inf.npy")
        path_Y = os.path.join(file_path, "unhealthy_inf.npy")

        if os.path.exists(path_X) and os.path.exists(path_Y):
            print("Loading influence scores from saved numpy files...")
            self.gr1_scr = np.load(path_X, mmap_mode="r")[: self.p]
            self.gr2_scr = np.load(path_Y, mmap_mode="r")[: self.q]
        else:
            raise FileNotFoundError(f"One or both influence score files not found: {path_X}, {path_Y}")

    def _initializer_groups(self):
        """
        Initialize the ranker with two groups of embeddings and their corresponding influence scores.
        """
        self._load_embeddings()
        self._load_infscores()

        assert len(self.gr1_embed) == len(self.gr1_scr), "Mismatch between group 1 embeddings and scores"
        assert len(self.gr2_embed) == len(self.gr2_scr), "Mismatch between group 2 embeddings and scores"

        # Combine scores and create unique IDs for tracking
        self.gr1_ids = np.arange(len(self.gr1_embed))
        self.gr2_ids = np.arange(len(self.gr2_embed))

    def _rank_samples(self):
        """
        Rank all samples in both groups based on influence scores in descending order.

        Returns:
            ranked_scores (np.ndarray): Influence scores sorted in descending order.
            ranked_ids (np.ndarray): Corresponding image IDs (group + index).
        """
        all_scores = np.concatenate([self.gr1_scr, self.gr2_scr])
        all_ids = np.concatenate([self.gr1_ids, self.gr2_ids + len(self.gr1_ids)])  # Offset group 2 IDs

        # Sort in descending order
        sorted_indices = np.argsort(-all_scores)
        ranked_scores = all_scores[sorted_indices]
        ranked_ids = all_ids[sorted_indices]

        # create a pandas dataframe to save ranked scores and ranked_ids
        df = pd.DataFrame({"ids": ranked_ids, "scores": ranked_scores})
        file_path = os.path.join(self.inf_dir, f"{self.seed}_{self.args.exp}_{self.args.ckp}_{self.n + self.m}")
        df_path = os.path.join(file_path, "df_rank.csv")

        if not os.path.exists(df_path):
            df.to_csv(df_path, index=False)

        return ranked_scores, ranked_ids

    def _exclude_top_percent(self, percent):
        """
        Exclude the top percentage of samples based on influence scores and return updated groups.

        Args:
            percent (float): Percentage of top samples to exclude (e.g., 1 for 1%).

        Returns:
            updated_gr1_embed (np.ndarray): Updated group 1 embeddings after exclusion.
            updated_gr2_embed (np.ndarray): Updated group 2 embeddings after exclusion.
            updated_gr1_scr (np.ndarray): Updated group 1 scores after exclusion.
            updated_gr2_scr (np.ndarray): Updated group 2 scores after exclusion.
        """
        total_samples = len(self.gr1_scr) + len(self.gr2_scr)
        exclude_count = int(np.ceil((percent / 100) * total_samples))
        print(f"{exclude_count} samples from {total_samples} with highest important scores were excluded!")

        # Rank all samples and get the IDs to exclude
        _, ranked_ids = self._rank_samples()
        ids_to_exclude = ranked_ids[:exclude_count]

        # Filter out embeddings and scores for group 1
        gr1_mask = np.isin(self.gr1_ids, ids_to_exclude, invert=True)

        updated_gr1_embed = self.gr1_embed[gr1_mask]
        updated_gr1_scr = self.gr1_scr[gr1_mask]

        # Filter out embeddings and scores for group 2
        gr2_mask = np.isin(self.gr2_ids, ids_to_exclude - len(self.gr1_ids), invert=True)
        updated_gr2_embed = self.gr2_embed[gr2_mask]
        updated_gr2_scr = self.gr2_scr[gr2_mask]

        return updated_gr1_embed, updated_gr2_embed, updated_gr1_scr, updated_gr2_scr

    def _compute_test_statistic(self, gr1_embed, gr2_embed):
        """
        Compute the test-statistic and p-value using the given embeddings.

        Args:
            gr1_embed (np.ndarray): Embeddings of group 1.
            gr2_embed (np.ndarray): Embeddings of group 2.

        Returns:
            test_statistic (float): The computed test-statistic.
            p_value (float): The p-value for the test.
        """
        mmdtest = MMDTest(gr1_embed, gr2_embed)
        p_value = mmdtest._compute_p_value()
        return p_value

    def run_analysis_top(self):
        """
        Perform the analysis by excluding top percentages and recomputing statistics.

        Args:
            self.args.excper (list): List of percentages to exclude.

        Returns:
            results (dict): Dictionary with keys as percentages and values as p-values.
        """
        dic_p = defaultdict()
        for perc in self.args.excper:
            updated_gr1_embed, updated_gr2_embed, _, _ = self._exclude_top_percent(perc)
            p_val = self._compute_test_statistic(updated_gr1_embed, updated_gr2_embed)
            dic_p[perc] = p_val
            print(f" p_val: {p_val}")

        df = pd.DataFrame([dic_p])
        # make directory, save results
        file_path = os.path.join(self.inf_dir, f"{self.seed}_{self.args.exp}_{self.args.ckp}_{self.n + self.m}")
        os.makedirs(file_path, exist_ok=True)
        max_per = self.args.excper[-1]
        df_path = os.path.join(file_path, f"df_p_most_{max_per}.csv")
        if not os.path.exists(df_path):
            df.to_csv(df_path, index=False)
        print(f"dic_p: {df}")

    def _exclude_low_percent(self, percent):
        """
        Exclude the bottom percentage of samples based on influence scores and return updated groups.

        Args:
            percent (float): Percentage of bottom samples to exclude (e.g., 1 for 1%).

        Returns:
            updated_gr1_embed (np.ndarray): Updated group 1 embeddings after exclusion.
            updated_gr2_embed (np.ndarray): Updated group 2 embeddings after exclusion.
            updated_gr1_scr (np.ndarray): Updated group 1 scores after exclusion.
            updated_gr2_scr (np.ndarray): Updated group 2 scores after exclusion.
        """
        total_samples = len(self.gr1_scr) + len(self.gr2_scr)
        exclude_count = int(np.ceil((percent / 100) * total_samples))
        print(f"{exclude_count} samples from {total_samples} with lowest important scores were excluded!")

        # Rank all samples and get the IDs to exclude
        _, ranked_ids = self._rank_samples()
        ids_to_exclude = ranked_ids[-exclude_count:]  # Select the bottom `exclude_count` samples

        # Filter out embeddings and scores for group 1
        gr1_mask = np.isin(self.gr1_ids, ids_to_exclude, invert=True)

        updated_gr1_embed = self.gr1_embed[gr1_mask]
        updated_gr1_scr = self.gr1_scr[gr1_mask]

        # Filter out embeddings and scores for group 2
        gr2_mask = np.isin(self.gr2_ids, ids_to_exclude - len(self.gr1_ids), invert=True)
        updated_gr2_embed = self.gr2_embed[gr2_mask]
        updated_gr2_scr = self.gr2_scr[gr2_mask]

        return updated_gr1_embed, updated_gr2_embed, updated_gr1_scr, updated_gr2_scr

    def run_analysis_low(self):
        """
        Perform the analysis by excluding percentages of lowest influential samples and recomputing statistics.

        Args:
            self.args.excper (list): List of percentages to exclude.

        Returns:
            results (dict): Dictionary with keys as percentages and values as p-values.
        """
        dic_p = defaultdict()
        for perc in self.args.excper:
            updated_gr1_embed, updated_gr2_embed, _, _ = self._exclude_low_percent(perc)
            p_val = self._compute_test_statistic(updated_gr1_embed, updated_gr2_embed)
            dic_p[perc] = p_val
            print(f" p_val: {p_val}")

        df = pd.DataFrame([dic_p])
        print(df)
        file_path = os.path.join(self.inf_dir, f"{self.seed}_{self.args.exp}_{self.args.ckp}_{self.n + self.m}")
        os.makedirs(file_path, exist_ok=True)
        max_per = self.args.excper[-1]
        df_path = os.path.join(file_path, f"df_p_least_{max_per}.csv")
        if not os.path.exists(df_path):
            df.to_csv(df_path, index=False)
        print(f"dic_p: {df}")


parser = argparse.ArgumentParser(description="Effect of excluding top/low important samples.")
parser.add_argument("--exp", type=str, default="sampleimp-fnt10-uncor", help="Experiment name")
parser.add_argument("--ckp", type=str, default="fnt", choices=["simclr", "fnt", "fnt_bl", "fnt_zer", "suppr"],
                    help="Checkpoint type")
parser.add_argument("--n", type=int, default=100, help="Number of samples in group 1")
parser.add_argument("--m", type=int, default=100, help="Number of samples in group 2")
parser.add_argument("--random_state", type=int, default=42, help="Random seed")
parser.add_argument("--excper", type=int, nargs="+", default=list(range(1, 50)),
                    help="The percentage of samples to exclude")
args = parser.parse_args()

if __name__ == "__main__":

    ranker = InfluenceRanker(args)
    ranker.run_analysis_top()
    ranker.run_analysis_low()
