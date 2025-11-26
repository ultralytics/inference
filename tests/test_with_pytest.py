# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from template.module1 import add_numbers, main


def test_add_numbers() -> None:
    """Tests the add_numbers function with positive and negative integer inputs.

    The function performs assertion tests to verify that the add_numbers function correctly handles both positive and
    negative integer addition operations. Tests include basic positive integer addition and operations involving
    negative integers.

    Examples:
        >>> test_add_numbers()  # Runs assertion tests
        >>> assert add_numbers(2, 3) == 5  # Test positive integers
        >>> assert add_numbers(-1, 1) == 0  # Test with negative integer
    """
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
    assert add_numbers(-1, -1) == -2


def test_main() -> None:
    """Tests main function executes without errors."""
    main()
