# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations


def add_numbers(a: int | float, b: int | float) -> int | float:
    """Adds two numbers together using element-wise addition.

    Args:
        a (int | float | torch.Tensor): First number or tensor to add.
        b (int | float | torch.Tensor): Second number or tensor to add.

    Returns:
        (int | float | torch.Tensor): Sum of the input numbers or tensors. If inputs are tensors, returns a tensor of
            the same shape.

    Examples:
        >>> result = add_numbers(1, 2)  # returns 3
        >>> x = torch.tensor([1, 2])
        >>> y = torch.tensor([3, 4])
        >>> result = add_numbers(x, y)  # returns tensor([4, 6])
    """
    return a + b


def main() -> None:
    """Main entry point for the Ultralytics Template application.

    Executes a simple addition operation by calling the add_numbers function with predefined values. This function
    serves as a basic demonstration of program flow and function calling within the Ultralytics framework.

    Examples:
        >>> main()  # Runs the main function which adds 1 + 2
    """
    a = 1
    b = 2
    y = add_numbers(a, b)
    print(f"Added {a} + {b} = {y}")


if __name__ == "__main__":
    main()
