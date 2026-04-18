import unittest

from tools import browser_tool


class BrowserScanToolTests(unittest.TestCase):
    def test_browser_scan_schema_is_registered(self):
        names = {schema["name"] for schema in browser_tool.BROWSER_TOOL_SCHEMAS}
        self.assertIn("browser_scan", names)

    def test_browser_scan_function_exists(self):
        self.assertTrue(callable(getattr(browser_tool, "browser_scan", None)))


if __name__ == "__main__":
    unittest.main()
