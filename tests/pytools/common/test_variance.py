from unittest import main, TestCase
import numpy as np
import numba as nb


@nb.njit(cache=True)
def two_pass_mean_sd(samples, n):
    meanloss = 0.0
    for l in samples:
        meanloss += l
    meanloss /= n
    if n != 1:
        sum_sq_dev = 0.0
        for l in samples:
            diff = l - meanloss
            sum_sq_dev += diff * diff
        variance = sum_sq_dev / (n - 1)
        sdloss = np.sqrt(variance)
    else:
        sdloss = 0.0
    return meanloss, sdloss


@nb.njit(cache=True)
def naive_mean_sd(samples, n):
    sumloss = 0.0
    sumlosssqr = 0.0
    for l in samples:
        sumloss += l
        sumlosssqr += l * l
    meanloss = sumloss / n
    if n != 1:
        variance = (sumlosssqr - (sumloss * sumloss) / n) / (n - 1)
        sdloss = np.sqrt(variance)
    else:
        sdloss = 0.0
    return meanloss, sdloss


class TestVarianceCalculation(TestCase):

    def test_basic_variance(self):
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        meanloss, sdloss = two_pass_mean_sd(samples, len(samples))
        self.assertAlmostEqual(meanloss, np.mean(samples))
        self.assertAlmostEqual(sdloss, np.std(samples, ddof=1))

    def test_single_sample(self):
        samples = np.array([42.0])
        meanloss, sdloss = two_pass_mean_sd(samples, 1)
        self.assertAlmostEqual(meanloss, 42.0)
        self.assertEqual(sdloss, 0.0)

    def test_all_zeros(self):
        samples = np.zeros(100)
        meanloss, sdloss = two_pass_mean_sd(samples, len(samples))
        self.assertEqual(meanloss, 0.0)
        self.assertEqual(sdloss, 0.0)

    def test_identical_values(self):
        samples = np.full(1000, 1e8)
        meanloss, sdloss = two_pass_mean_sd(samples, len(samples))
        self.assertAlmostEqual(meanloss, 1e8)
        self.assertAlmostEqual(sdloss, 0.0)

    def test_catastrophic_cancellation(self):
        rng = np.random.default_rng(12345)
        samples = 1e8 + rng.normal(0, 1, 1000)
        n = len(samples)

        expected_sd = np.std(samples, ddof=1)

        _, sd_two_pass = two_pass_mean_sd(samples, n)
        _, sd_naive = naive_mean_sd(samples, n)

        rel_err_two_pass = abs(sd_two_pass - expected_sd) / expected_sd
        rel_err_naive = abs(sd_naive - expected_sd) / expected_sd

        self.assertLess(rel_err_two_pass, 1e-10)
        self.assertGreater(rel_err_naive, rel_err_two_pass)

    def test_mixed_zero_nonzero(self):
        samples = np.array([0.0, 0.0, 0.0, 100.0, 200.0])
        meanloss, sdloss = two_pass_mean_sd(samples, len(samples))
        self.assertAlmostEqual(meanloss, np.mean(samples))
        self.assertAlmostEqual(sdloss, np.std(samples, ddof=1))

    def test_large_sample_count(self):
        rng = np.random.default_rng(99)
        samples = rng.exponential(scale=5000.0, size=10000)
        meanloss, sdloss = two_pass_mean_sd(samples, len(samples))
        self.assertAlmostEqual(meanloss, np.mean(samples), places=5)
        self.assertAlmostEqual(sdloss, np.std(samples, ddof=1), places=5)


if __name__ == "__main__":
    main()
