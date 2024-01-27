import unittest
from optimizers import ldp_mechanism
import torch

class Test(unittest.TestCase):

    def testLdp(self):
        test_tensor = [torch.ones(20)]
        result = ldp_mechanism(test_tensor, 1, 100)
        print(result)



if __name__ == '__main__':
    unittest.main()