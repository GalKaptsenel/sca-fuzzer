"""
File: various ways to compare ctraces with htraces

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import hashlib
from cachetools import LRUCache
from joblib import Parallel, delayed

from collections import defaultdict, Counter, OrderedDict
from typing import List, Dict, Tuple, Optional, Callable
from scipy import stats  # type: ignore

from .interfaces import HTrace, CTrace, Input, EquivalenceClass, Analyser, Measurement, Violation, TestCase
from .config import CONF
from .util import STAT, Logger


class EquivalenceAnalyserCommon(Analyser):

    def __init__(self) -> None:
        self.LOG = Logger()
        super().__init__()

    def filter_violations(self,
                          inputs: List[Input],
                          ctraces: List[CTrace],
                          htraces: List[HTrace],
                          stats=False,
			  test_cases=None) -> List[Violation]:
        """
        Group the measurements by their ctrace (i.e., build equivalence classes of measurements
        w.r.t. their ctrace) and check if all htraces in the same equivalence class are equal.

        Note that the htraces are not necessarily compared directly (i.e., we don't always
        check if htrace1 == htrace2). This isn't always possible because the measurements
        are noisy, and we need to allow for some differences. Instead, each of the subclasses
        of this class implements a different way to compare the htraces. For example, the
        ProbabilisticAnalyser compares the distributions of traces.

        :param inputs: a list of inputs
        :param ctraces: a list of contract traces
        :param htraces: a list of hardware traces
        :param stats: whether to update the statistics based on the results
        :return: if a violation is found, return a list of equivalence classes that contain
                 contract counterexamples. Otherwise, return an empty list.
        """
        # Skip if there are no htraces
        if not htraces:
            return []

        # Build a list of equivalence classes:
        #   1. Map ctraces to their IDs
        equivalent_inputs_ids = defaultdict(list)
        for i, ctrace in enumerate(ctraces):
            # skip the measurements with corrupted/ignored htraces
            if not htraces[i].raw:
                 continue
            equivalent_inputs_ids[ctrace].append(i)
        

        #   2. Build equivalence classes
        effective_classes: List[EquivalenceClass] = []
        for ctrace, ids in equivalent_inputs_ids.items():
            # skip ineffective eq. classes
            if len(ids) < 2:
                continue

            # get all measurements in the class
            if test_cases:
                measurements = [Measurement(i, inputs[i], ctrace, htraces[i], test_cases[i]) for i in ids]
            else:
                measurements = [Measurement(i, inputs[i], ctrace, htraces[i]) for i in ids]
             

            # Build htrace groups
            htrace_groups = self._build_htrace_groups(measurements)

            # Create an equivalence class
            eq_cls = EquivalenceClass(ctrace, measurements, htrace_groups)
            effective_classes.append(eq_cls)

        #   3. Sort the equivalence classes by ctrace
        effective_classes.sort(key=lambda x: x.ctrace)

        # Check if any of the equivalence classes is a contract counterexample
        violations: List[Violation] = []
        for eq_cls in effective_classes:
            if len(eq_cls.htrace_groups) >= 2:
                violations.append(Violation(eq_cls, inputs))

        # Update statistics
        if stats:
            STAT.eff_classes += len(effective_classes)
            STAT.single_entry_classes += len(equivalent_inputs_ids) - len(effective_classes)
            STAT.analysed_test_cases += 1

        return violations

    def _build_htrace_groups(self, measurements: List[Measurement]) -> List[List[Measurement]]:
        """
        Group measurements that have equivalent htraces, and set the htrace_groups attribute
        for the given equivalence class

        :param measurements: List of measurements to be grouped
        :return: List of groups of measurements
        """
        groups: List[List[Measurement]] = []
        for m in measurements:
            if not groups:
                groups.append([m])
                continue

            for group in groups:
                if self.htraces_are_equivalent(m.htrace, group[0].htrace):
                    group.append(m)
                    break
            else:
                groups.append([m])
        return groups


class MergedBitmapAnalyser(EquivalenceAnalyserCommon):
    """ A variant of the analyser that compares the htraces as merged bitmaps. I.e., it merges
    the htrace lists into bitmaps and compares the results.

    It also applies filtering of outliers according to CONF.analyser_outliers_threshold
    """

    bitmap_cache: Dict[int, int]
    MASK = pow(2, 64) - 1

    def __init__(self):
        self.bitmap_cache = {}

    def htraces_are_equivalent(self, htrace1: HTrace, htrace2: HTrace) -> bool:
        bitmaps = [0, 0]
        sample_size = len(htrace1.raw)
        assert sample_size == len(htrace2.raw), "htraces have different sizes"
        threshold = CONF.analyser_outliers_threshold * sample_size
        for i, htrace in enumerate([htrace1, htrace2]):
            # check if cached
            if htrace.hash_ in self.bitmap_cache:
                bitmaps[i] = self.bitmap_cache[htrace.hash_]
                continue

            # remove outliers
            counter = Counter(htrace.raw)
            filtered = [x for x in htrace.raw if counter[x] >= threshold]

            # merge into bitmap
            for t in filtered:
                bitmaps[i] |= t

            # cache
            self.bitmap_cache[htrace.hash_] = bitmaps[i]

        if CONF.analyser_subsets_is_violation:
            return bitmaps[0] == bitmaps[1]

        # check if the bitmaps are disjoint
        inverse = [~bitmaps[0] & self.MASK, ~bitmaps[1] & self.MASK]
        return (bitmaps[0] & inverse[1]) == 0 or (bitmaps[1] & inverse[0]) == 0


class SetAnalyser(EquivalenceAnalyserCommon):
    """ A variant of the analyser that compares the htraces as sets. I.e., it squashes
    the htrace lists into sets and compares the results.

    It also applies filtering of outliers according to CONF.analyser_outliers_threshold
    """

    def htraces_are_equivalent(self, htrace1: HTrace, htrace2: HTrace) -> bool:
        """ Squash the htrace lists into sets and compare the results """
        sample_size = len(htrace1.raw)
        assert sample_size == len(htrace2.raw), "htraces have different sizes"
        threshold = CONF.analyser_outliers_threshold * sample_size
        filtered1 = [x for x in htrace1.raw if x >= threshold]
        filtered2 = [x for x in htrace2.raw if x >= threshold]

        trace_set1 = set(filtered1)
        trace_set2 = set(filtered2)

        if CONF.analyser_subsets_is_violation:
            return trace_set1 == trace_set2

        return trace_set1.issubset(trace_set2) or trace_set2.issubset(trace_set1)


class MWUAnalyser(EquivalenceAnalyserCommon):
    """ A variant of the analyser that uses the Mann-Withney U test to compare htraces.

    WARNING: this is an experimental analyser and it may not work well for all cases."""
    last_p_value: float = 0.0

    def __init__(self) -> None:
        super().__init__()
        self.LOG.warning(
            "analyser",
            "MWUAnalyser is an experimental analyser and may not work well for all cases. ")

        a = [1] * CONF.executor_sample_sizes[0]
        b = [2] * CONF.executor_sample_sizes[0]
        _, p_value = stats.mannwhitneyu(a, b)
        if CONF.analyser_stat_threshold < p_value:
            self.LOG.error("analyser_stat_threshold is too low for the given sample size")

    def htraces_are_equivalent(self, htrace1: HTrace, htrace2: HTrace) -> bool:
        """ Use the Mann-Withney U test to compare htraces """
        _, p_value = stats.mannwhitneyu(htrace1.raw, htrace2.raw)

        # print(set(htrace1.raw), set(htrace2.raw), p_value)
        # if p_value <= CONF.analyser_stat_threshold:
        # print(f"p_value={p_value:.6f}")
        return p_value > CONF.analyser_stat_threshold


class ChiSquaredAnalyser(EquivalenceAnalyserCommon):

    def __init__(self) -> None:
        super().__init__()
        a = [1] * CONF.executor_sample_sizes[0]
        b = [2] * CONF.executor_sample_sizes[0]
        stat = self.homogeneity_test(a, b)
        if CONF.analyser_stat_threshold > stat:
            self.LOG.error("analyser_stat_threshold is too low for the given sample size")

    def homogeneity_test(self, x: List[int], y: List[int]) -> bool:
        """ Use the chi-squared test to compare htraces """
        assert len(x) == len(y)
        counter1 = Counter(x)
        counter2 = Counter(y)
        keys = set(counter1.keys()) | set(counter2.keys())
        observed = [counter1[k] for k in keys] + [counter2[k] for k in keys]
        expected = [(counter1[k] + counter2[k]) / 2 for k in keys] * 2
        ddof = len(keys) - 1
        stat, _ = stats.chisquare(observed, expected, ddof=ddof)
        stat /= len(x) + len(y)
        return stat

    def htraces_are_equivalent(self, htrace1: HTrace, htrace2: HTrace) -> bool:
        stat = self.homogeneity_test(htrace1.raw, htrace2.raw)
        return stat < CONF.analyser_stat_threshold

class ChiSquaredBitwisePValueAnalyser(EquivalenceAnalyserCommon):
    _corr_cache = LRUCache(maxsize=128)
    _alpha = 0.2

    def __init__(self) -> None:
        super().__init__()
        a = [np.uint64(random.getrandbits(64)) for _ in range(CONF.executor_sample_sizes[0])]
        b = [np.uint64(random.getrandbits(64)) for _ in range(CONF.executor_sample_sizes[0])]

        if self.homogeneity_test(a, b, lambda x: self._alpha > x):
            self.LOG.error("Default inputs yield a statistically significant difference. Consider increasing sample size or investigating test setup.")

    @classmethod
    def _array_hash(cls, arr: np.ndarray) -> str:
        # Compute a hash string of the array bytes for cache key
        return hashlib.sha256(arr.tobytes()).hexdigest()

    @staticmethod
    def traces_to_bit_matrix(traces: List[np.uint64]) -> np.ndarray:
        arr = np.array(traces, dtype=np.uint64)
        bits = ((arr[:, None] >> np.arange(64, dtype=np.uint64)) & 1).astype(np.uint8)
        return bits

    @classmethod
    def bitwise_pvalue_distribution(cls, bits_a: np.ndarray, bits_b: np.ndarray) -> Tuple[float, List[float]]:
        p_values = []

        for bit in range(64):
            counts = np.array([
                [np.count_nonzero(bits_a[:, bit] == 0), np.count_nonzero(bits_a[:, bit] == 1)],
                [np.count_nonzero(bits_b[:, bit] == 0), np.count_nonzero(bits_b[:, bit] == 1)],
                ])

            if np.any(counts.sum(axis=0) == 0) or np.any(counts.sum(axis=1) == 0):
                # Append a neutral p-value (e.g., 1.0) since no difference can be detected
                p_values.append(1.0)
                continue

            _, p, _, _ = stats.chi2_contingency(counts)
            p_values.append(p)

        _, combined_p = cls.safe_combine_pvalues(p_values, method='fisher')
        return combined_p, p_values

    @staticmethod
    def safe_combine_pvalues(pvalues, method='fisher', eps=1e-300):
        pvalues = np.clip(pvalues, eps, 1.0)
        return stats.combine_pvalues(pvalues, method=method)

    @classmethod
    def correlation_pvalue_parallel(cls, bits_a: np.ndarray, bits_b: np.ndarray, n_permutations: int = 1000, n_jobs: int = -1) -> float:

        def single_permutation(all_bits, n, observed_diff):
            permuted = np.copy(all_bits)
            np.random.shuffle(permuted)
            perm_a = permuted[:n]
            perm_b = permuted[n:]
    
            diff = np.linalg.norm(
                cls.bitwise_correlation_matrix(perm_a) - cls.bitwise_correlation_matrix(perm_b),
                ord='fro'
            )

            return diff >= observed_diff

        corr_a = cls.bitwise_correlation_matrix(bits_a)
        corr_b = cls.bitwise_correlation_matrix(bits_b)
        observed_diff = np.linalg.norm(corr_a - corr_b, ord='fro')

        all_bits = np.concatenate([bits_a, bits_b], axis=0)
        n = bits_a.shape[0]

        results = Parallel(n_jobs=n_jobs)(delayed(single_permutation)(all_bits, n, observed_diff) for _ in range(n_permutations))
        count = sum(results)
        return (count + 1) / (n_permutations + 1)

    @classmethod
    def correlation_pvalue(cls, bits_a: np.ndarray, bits_b: np.ndarray, n_permutations: int = 1000) -> float:
        corr_a = cls.bitwise_correlation_matrix(bits_a)
        corr_b = cls.bitwise_correlation_matrix(bits_b)
        observed_diff = np.linalg.norm(corr_a - corr_b, ord='fro')

        all_bits = np.concatenate([bits_a, bits_b], axis=0)
        n = bits_a.shape[0]

        count = 0
        for _ in range(n_permutations):
            permuted = np.copy(all_bits)
            np.random.shuffle(permuted)
            perm_a = permuted[:n]
            perm_b = permuted[n:]
            diff = np.linalg.norm(
                    cls.bitwise_correlation_matrix(perm_a) - cls.bitwise_correlation_matrix(perm_b),
                    ord='fro'
                    )
            if diff >= observed_diff:
                count += 1

        return (count + 1) / (n_permutations + 1)

    @staticmethod
    def safe_corrcoef(bits: np.ndarray, noise_threshold: float = 1e-5) -> np.ndarray:
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = np.corrcoef(bits.T)

        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = np.clip(corr, -1.0, 1.0)
        corr[np.abs(corr) < noise_threshold] = 0.0

        return corr

    @classmethod
    def bitwise_correlation_matrix(cls, bits: np.ndarray) -> np.ndarray:
        key = cls._array_hash(bits)
        cached = cls._corr_cache.get(key)
        if cached is not None:
            return cached

        corr = cls.safe_corrcoef(bits)
        cls._corr_cache[key] = corr
        return corr

    @staticmethod
    def correlation_difference(corr_a: np.ndarray, corr_b: np.ndarray) -> float:
        return np.linalg.norm(corr_a - corr_b, ord='fro')

    @classmethod
    def visualize_bitwise_and_correlation(cls, bits_a: np.ndarray, bits_b: np.ndarray, save_path: Optional[str] = None) -> None:

        _, p_values = cls.bitwise_pvalue_distribution(bits_a, bits_b)

        corr_a = cls.bitwise_correlation_matrix(bits_a)
        corr_b = cls.bitwise_correlation_matrix(bits_b)
        corr_diff = np.abs(corr_a - corr_b)

        fig, axs = plt.subplots(2, 1, figsize=(12, 10))

        axs[0].bar(range(64), p_values, color='blue', alpha=0.7)
        axs[0].axhline(0.05, color='red', linestyle='--', label=f'Significance threshold ({cls._alpha})')
        axs[0].set_title("Bitwise Chi-Squared Test p-values per Bit")
        axs[0].set_xlabel("Bit Position")
        axs[0].set_ylabel("p-value")
        axs[0].set_ylim(0, 1)
        axs[0].legend()
        axs[0].set_xticks(range(0, 64, 4))

        im = axs[1].imshow(corr_diff, cmap='hot', interpolation='nearest')
        axs[1].set_title("Absolute Difference of Bitwise Correlation Matrices")
        axs[1].set_xlabel("Bit Position")
        axs[1].set_ylabel("Bit Position")
        axs[1].set_xticks(range(0, 64, 4))
        axs[1].set_yticks(range(0, 64, 4))
        axs[1].set_xticklabels(range(0, 64, 4))
        axs[1].set_yticklabels(range(0, 64, 4))
        fig.colorbar(im, ax=axs[1])

        plt.tight_layout()

        if save_path is not None:
            #print("Saving plot")
            plt.savefig(save_path)

        plt.close(fig)
#        plt.show()

    def homogeneity_test(self, x: List[np.uint64], y: List[np.uint64], pass_test: Callable[[float], bool], visualize: bool = False) -> bool:
        assert len(x) == len(y)

        if len(x) < 10:
            self.LOG.warning("Sample size may be too small for reliable p-value analysis.")

        bits_x = self.traces_to_bit_matrix(x)
        bits_y = self.traces_to_bit_matrix(y)
        
        p_dist, _ = self.bitwise_pvalue_distribution(bits_x, bits_y)
        p_corr = self.correlation_pvalue(bits_x, bits_y, n_permutations=200)
        _, final_p = self.safe_combine_pvalues([p_dist, p_corr], method='fisher')

        result = pass_test(final_p)
        if visualize and not result:
            self.visualize_bitwise_and_correlation(bits_x, bits_y, "bitwise_pvalues_and_corr_diff.png")
            #print(f"Bitwise p = {p_dist:.4g}, Corr p = {p_corr:.4g}, Final combined p = {final_p:.4g}")

        return result

    def htraces_are_equivalent(self, htrace1: HTrace, htrace2: HTrace) -> bool:
        return self.homogeneity_test(htrace1.raw, htrace2.raw, pass_test=lambda x: x > self._alpha, visualize=False)


class ChiSquaredBitwiseThresholdAnalyser(EquivalenceAnalyserCommon):
    _corr_cache = LRUCache(maxsize=128)
    _threshold = 0.01

    def __init__(self) -> None:
        super().__init__()
        a = [np.uint64(1)] * CONF.executor_sample_sizes[0]
        b = [np.uint64(2)] * CONF.executor_sample_sizes[0]
        if self.homogeneity_test(a, b, lambda x: self._threshold > x):
            self.LOG.error("analyser_stat_threshold is too low for the given sample size")

    @classmethod
    def _array_hash(cls, arr: np.ndarray) -> str:
        return hashlib.sha256(arr.tobytes()).hexdigest()

    @staticmethod
    def traces_to_bit_matrix(traces: List[np.uint64]) -> np.ndarray:
        arr = np.array(traces, dtype=np.uint64)
        bits = ((arr[:, None] >> np.arange(64, dtype=np.uint64)) & 1).astype(np.uint8)
        return bits

    @staticmethod
    def bitwise_distribution_difference(bits_a: np.ndarray, bits_b: np.ndarray) -> float:
        stat_sum = 0
        number_bits_considered = 0

        for bit in range(64):
            counts = np.array([
                [np.count_nonzero(bits_a[:, bit] == 0), np.count_nonzero(bits_a[:, bit] == 1)],
                [np.count_nonzero(bits_b[:, bit] == 0), np.count_nonzero(bits_b[:, bit] == 1)],
                ])

            if np.any(counts.sum(axis=0) == 0) or np.any(counts.sum(axis=1) == 0):
                continue

            stat, _, _, _ = stats.chi2_contingency(counts)
            stat_sum += stat
            number_bits_considered += 1

        if number_bits_considered == 0:
            return 0.0

        return stat_sum / number_bits_considered


    @staticmethod
    def safe_corrcoef(bits: np.ndarray, noise_threshold: float = 1e-5) -> np.ndarray:
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = np.corrcoef(bits.T)

        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = np.clip(corr, -1.0, 1.0)
        corr[np.abs(corr) < noise_threshold] = 0.0

        return corr

    @classmethod
    def bitwise_correlation_matrix(cls, bits: np.ndarray) -> np.ndarray:
        key = cls._array_hash(bits)
        cached = cls._corr_cache.get(key)
        if cached is not None:
            return cached

        corr = cls.safe_corrcoef(bits)
        cls._corr_cache[key] = corr
        return corr

    @staticmethod
    def correlation_difference(corr_a: np.ndarray, corr_b: np.ndarray) -> float:
        return np.linalg.norm(corr_a - corr_b, ord='fro')

    @classmethod
    def bitwise_pvalue_distribution(cls, bits_a: np.ndarray, bits_b: np.ndarray) -> Tuple[float, List[float]]:
        p_values = []
        for bit in range(64):
            counts = np.array([
                [np.count_nonzero(bits_a[:, bit] == 0), np.count_nonzero(bits_a[:, bit] == 1)],
                [np.count_nonzero(bits_b[:, bit] == 0), np.count_nonzero(bits_b[:, bit] == 1)],
            ])

            if np.any(counts.sum(axis=0) == 0) or np.any(counts.sum(axis=1) == 0):
                p_values.append(1.0)
                continue

            _, p, _, _ = stats.chi2_contingency(counts)
            p_values.append(p)

        _, combined_p = cls.safe_combine_pvalues(p_values, method='fisher')
        return combined_p, p_values

    @staticmethod
    def safe_combine_pvalues(pvalues, method='fisher', eps=1e-300):
        pvalues = np.clip(pvalues, eps, 1.0)
        return stats.combine_pvalues(pvalues, method=method)

    @classmethod
    def visualize_bitwise_and_correlation(
        cls,
        bits_a: np.ndarray,
        bits_b: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        _, p_values = cls.bitwise_pvalue_distribution(bits_a, bits_b)

        corr_a = cls.bitwise_correlation_matrix(bits_a)
        corr_b = cls.bitwise_correlation_matrix(bits_b)
        corr_diff = np.abs(corr_a - corr_b)

        fig, axs = plt.subplots(2, 1, figsize=(12, 10))

        axs[0].bar(range(64), p_values, color='blue', alpha=0.7)
        axs[0].axhline(0.05, color='red', linestyle='--', label=f'Significance threshold ({cls._threshold})') # TODO: This is incorrect labels!
        axs[0].set_title("Bitwise Chi-Squared Test p-values per Bit")
        axs[0].set_xlabel("Bit Position")
        axs[0].set_ylabel("p-value")
        axs[0].set_ylim(0, 1)
        axs[0].legend()

        im = axs[1].imshow(corr_diff, cmap='hot', interpolation='nearest')
        axs[1].set_title("Absolute Difference of Bitwise Correlation Matrices")
        axs[1].set_xlabel("Bit Position")
        axs[1].set_ylabel("Bit Position")
        fig.colorbar(im, ax=axs[1])

        plt.tight_layout()

        if save_path is not None:
            #print(f"Saving plot to {save_path}")
            plt.savefig(save_path)

        plt.close(fig)

    def homogeneity_test(self, x: List[np.uint64], y: List[np.uint64], pass_test: Callable[[float], bool], visualize: bool = False) -> bool:
        assert len(x) == len(y)
        bits_x = self.traces_to_bit_matrix(x)
        bits_y = self.traces_to_bit_matrix(y)
        
        normalized_bitwise_stat = self.bitwise_distribution_difference(bits_x, bits_y)
        corr_diff = self.correlation_difference(
                self.bitwise_correlation_matrix(bits_x),
                self.bitwise_correlation_matrix(bits_y)
                )
        NUM_BITS = 64
        max_corr_diff = 2 * NUM_BITS
        normalized_corr_diff = corr_diff / max_corr_diff

        weight = 0.7
        combined_stat = (weight * normalized_bitwise_stat + (1 - weight) * normalized_corr_diff)

        result = pass_test(combined_stat)
        if visualize and not result:
            self.visualize_bitwise_and_correlation(bits_x, bits_y, "bitwise_threshlod.png")
            #print(f"[DEBUG] bitwise_stat={normalized_bitwise_stat:.4f}, corr_diff={corr_diff:.4f}, combined_stat={combined_stat:.4f}")

        return result

    def htraces_are_equivalent(self, htrace1: HTrace, htrace2: HTrace) -> bool:
        return self.homogeneity_test(htrace1.raw, htrace2.raw, pass_test=lambda x: x < self._threshold, visualize=False)


class AnsambleAnalyser(EquivalenceAnalyserCommon):
    def __init__(self):
        self.analysers: Dict[str, EquivalenceAnalyserCommon] = OrderedDict({
                'ChiSquaredAnalyser': ChiSquaredAnalyser(),
                'ChiSquaredBitwisePValueAnalyser': ChiSquaredBitwisePValueAnalyser(),
#                'ChiSquaredBitwiseThresholdAnalyser': ChiSquaredBitwiseThresholdAnalyser(),
        })
        self.scores = defaultdict(int)
        self.experiments_counter = 0
        self.reports = []

    def htraces_are_equivalent(self, htrace1: HTrace, htrace2: HTrace) -> bool:
        votes = 0
        print(f"htrace1:")
        for n in htrace1.raw:
            print(f"{format(n, '064b')}")
        print(f"htrace2:")
        for n in htrace2.raw:
            print(f"{format(n, '064b')}")


        for name, analyser in self.analysers.items():
            if analyser.htraces_are_equivalent(htrace1, htrace2):
                self.scores[name] += 1
                votes += 1
#                print(f'Analyser {name} claims {htrace1.raw} and {htrace2.raw} are equivalent')
#            else:
#                print(f'Analyser {name} claims {htrace1.raw} and {htrace2.raw} are distinct')


        self.experiments_counter += 1

        return votes >= (len(self.analysers) / 2)

    def _log_scores(self, equivalent: bool):
        report = self.summary(equivalent)
        print(report)
        return report

    def summary(self, equivalent: bool):
        def format_score(name):
            equivalent_count = self.scores[name]
            percentage = (equivalent_count / self.experiments_counter) if self.experiments_counter > 0 else 0
            if not equivalent:
                percentage = 1 - percentage
                equivalent_count = self.experiments_counter - equivalent_count
            return f'{name}: {equivalent_count}/{self.experiments_counter} ({percentage * 100:.2f}%)'

        scores_str = '\t'.join(format_score(name) for name in self.analysers)
        return f'[AnsambleAnalyser] Scores: {scores_str}'

    def reset(self, equivalent: bool): 
        prefix = "Violation Detected: " if not equivalent else ""
        self.reports.append(prefix + self._log_scores(equivalent))
        self.scores = defaultdict(int)
        self.experiments_counter = 0

    def __del__(self):
        try:
            print('\n[AnsambleAnalyser] Final Report:')
            for report in self.reports:
                print(report)
        except Exception:
            pass
