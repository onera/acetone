import unittest
#import acetone_nnet
from acetone_nnet.quantize.qform import parse_q_format
# --- Unit Tests ---

class TestParseQFormat(unittest.TestCase):
    """
    Test suite for the parse_q_format function.
    """

    def test_valid_format(self):
        """Tests parsing of valid Q format strings."""
        self.assertEqual(parse_q_format("Q15.16"), (15, 16))
        self.assertEqual(parse_q_format("Q3.28"), (3, 28))
        self.assertEqual(parse_q_format("Q0.8"), (0, 8))
        self.assertEqual(parse_q_format("Q1.0"), (1, 0))

    def test_invalid_prefix(self):
        """Tests strings with an incorrect prefix."""
        self.assertIsNone(parse_q_format("A15.16"))
        self.assertIsNone(parse_q_format("q15.16")) # Case-sensitive

    def test_missing_parts(self):
        """Tests strings that are missing parts of the format."""
        self.assertIsNone(parse_q_format("Q15"))
        self.assertIsNone(parse_q_format("Q.16"))
        self.assertIsNone(parse_q_format("Q15."))

    def test_non_digit_characters(self):
        """Tests strings containing non-digit characters for x or y."""
        self.assertIsNone(parse_q_format("Q15.a16"))
        self.assertIsNone(parse_q_format("Qx.16"))
        self.assertIsNone(parse_q_format("Q15.16b"))

    def test_empty_and_malformed_strings(self):
        """Tests various other malformed strings."""
        self.assertIsNone(parse_q_format(""))
        self.assertIsNone(parse_q_format("Q."))
        self.assertIsNone(parse_q_format("Not a Q string"))


# This allows the test suite to be run from the command line.
if __name__ == '__main__':
    # Using argv and exit=False makes it compatible with some environments
    # that might not handle sys.exit() well.
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
