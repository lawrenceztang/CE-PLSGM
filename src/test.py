import unittest
from optimizers import ldp_mechanism
import torch
from train import find_epsilon
import math

class Test(unittest.TestCase):

    def testLdp(self):
        test_tensor = [torch.ones(20)]
        result = ldp_mechanism(test_tensor, 1, 3)
        print(result)

    def test_find_epsilon(self):
        eps0 = 5
        k = 1000
        delta = 10**(-5)
        eps = find_epsilon(eps0, k, delta)
        print("Epsilon:", eps)
        print("Epsilon0:", eps / (2 * math.sqrt(k * math.log(math.e + eps / delta))))


if __name__ == '__main__':
    unittest.main()