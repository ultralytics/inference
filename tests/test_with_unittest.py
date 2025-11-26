# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import unittest
from template.module1 import add_numbers, main

class TestModule1(unittest.TestCase):
    """Test cases for module1 functions."""
    
    def test_add_numbers(self) -> None:
        """Tests add_numbers function with positive and negative integers."""
        self.assertEqual(add_numbers(2, 3), 5)
        self.assertEqual(add_numbers(-1, 1), 0)
        self.assertEqual(add_numbers(-1, -1), -2)

    def test_main(self) -> None:
        """Tests main function executes without errors."""
        main()

if __name__ == "__main__":
    unittest.main()
