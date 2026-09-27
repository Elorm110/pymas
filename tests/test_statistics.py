import unittest
from pymas.statistics import mean, median, mode, variance, sample_variance, percentile, correlation, linear_regression, probability, normal_pdf, binomial, normalize, standardize

class TestPymas(unittest.TestCase):
    def test_mean(self): self.assertEqual(mean([1,2,3,4,5]),3)
    def test_median(self): self.assertEqual(median([1,2,3,4]),2.5)
    def test_mode(self): self.assertEqual(mode([1,2,2,3]),2)
    def test_variance(self): self.assertAlmostEqual(variance([1,2,3]),2/3)
    def test_sample_variance(self): self.assertAlmostEqual(sample_variance([1,2,3]),1)
    def test_percentile(self): self.assertEqual(percentile([1,2,3,4,5],50),3)
    def test_correlation(self): self.assertAlmostEqual(correlation([1,2,3],[2,4,6]),1)
    def test_regression(self):
        r=linear_regression([1,2,3],[2,4,6]); self.assertAlmostEqual(r["slope"],2); self.assertAlmostEqual(r["intercept"],0)
    def test_probability(self): self.assertEqual(probability(2,4),0.5)
    def test_normal_pdf(self): self.assertAlmostEqual(normal_pdf(0,0,1),0.39894228,places=6)
    def test_binomial(self): self.assertAlmostEqual(binomial(4,0.5,2),0.375)
    def test_normalize(self): self.assertEqual(normalize([10,20,30]),[0.0,0.5,1.0])
    def test_standardize(self): self.assertAlmostEqual(standardize([1,2,3])[1],0)

if __name__=="__main__": unittest.main()
