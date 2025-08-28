import re


def parse_q_format(q_string: str) -> tuple[int, int] | None:
    """
    Parses a string in the Qx.y format using a regular expression.

    The Q format is a fixed-point number representation. "x" represents the
    number of integer bits, and "y" represents the number of fractional bits.

    Args:
        q_string: The string to parse, expected in "Qx.y" format.

    Returns:
        A tuple containing the two integers (x, y) if the format is valid.
        Returns None if the input string does not match the expected format.
    """
    # Define the regular expression pattern.
    # - ^ asserts position at start of the string.
    # - Q matches the literal character 'Q'.
    # - (\d+) captures one or more digits. This is the first capturing group (for x).
    # - \. matches the literal character '.'. The backslash escapes the dot,
    #   which is a special character in regex (matches any character).
    # - (\d+) captures one or more digits. This is the second capturing group (for y).
    # - $ asserts position at the end of the string.
    pattern = r"^Q(-?\d+)\.(\d+)$"

    # Use re.match() to find a match at the beginning of the string.
    match = re.match(pattern, q_string)

    # Check if a match was found.
    if match:
        # The match object's groups() method returns a tuple of all captured groups.
        # group(1) corresponds to the first (\d+), which is 'x'.
        # group(2) corresponds to the second (\d+), which is 'y'.
        x_str, y_str = match.groups()

        # Convert the extracted string parts to integers.
        x = int(x_str)
        y = int(y_str)

        return (x, y)
    else:
        # If the pattern does not match, return None.
        return None
