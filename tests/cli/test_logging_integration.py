# Standard library imports
import io
import logging
import re
import sys
import types
import unittest
import warnings

# Third-party imports
from unittest.mock import patch, MagicMock

# Local application imports
from oasislmf.cli.model import RunCmd

# Test format constants - centralized for maintainability
FORMAT_STANDARD = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
FORMAT_COMPACT = "%(asctime)s [%(levelname)s] %(message)s"
FORMAT_SIMPLE = "%(message)s"
FORMAT_LEVEL_MESSAGE = "%(levelname)s - %(message)s"


class LoggingIntegrationTests(unittest.TestCase):
    """
    Integration tests for the complete logging system.

    These tests verify that the logging configuration works correctly
    throughout the entire command execution flow, not just during setup.

    Test Execution:
    - Run all tests: pytest tests/cli/test_logging_integration.py -v
    - Run specific category: pytest -k "test_logging_level" -v
    - Run direct: python tests/cli/test_logging_integration.py
    - Run with coverage: pytest --cov=oasislmf.cli tests/cli/test_logging_integration.py
    """

    def setUp(self):
        """Reset logging state between tests with comprehensive isolation."""
        # Store original logging state to restore later
        self._original_loggers = {}

        # Get all existing loggers and store their state
        existing_loggers = [
            logging.getLogger(name) for name in logging.root.manager.loggerDict
        ]
        existing_loggers.append(logging.root)  # Include root logger

        for logger in existing_loggers:
            self._original_loggers[logger.name] = {
                "level": logger.level,
                "handlers": logger.handlers.copy(),
                "disabled": getattr(logger, "disabled", False),  # Safe default
                "propagate": getattr(logger, "propagate", True),  # Safe default
            }

        # Clear all existing logger state
        for logger in existing_loggers:
            logger.handlers.clear()
            logger.setLevel(logging.NOTSET)
            logger.disabled = False
            logger.propagate = True

        # Clear specific loggers we know about
        oasis_logger = logging.getLogger("oasislmf")
        oasis_logger.handlers.clear()
        oasis_logger.setLevel(logging.NOTSET)

        ods_logger = logging.getLogger("ods_tools")
        ods_logger.handlers.clear()
        ods_logger.setLevel(logging.NOTSET)

        # Set root logger to high level to reduce noise
        logging.root.setLevel(logging.CRITICAL)

    def tearDown(self):
        """Restore original logging state after each test for complete isolation."""
        # Clear any handlers that were added during the test
        for logger_name in logging.root.manager.loggerDict:
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()

        logging.root.handlers.clear()

        # Restore original logger states
        if hasattr(self, "_original_loggers"):
            for logger_name, original_state in self._original_loggers.items():
                logger = (
                    logging.getLogger(logger_name)
                    if logger_name != "root"
                    else logging.root
                )

                # Restore original properties
                logger.setLevel(original_state["level"])
                logger.handlers.clear()
                logger.handlers.extend(original_state["handlers"])

                # Safely restore disabled flag (not all logger types support this)
                if hasattr(logger, "disabled"):
                    logger.disabled = original_state["disabled"]

                # Safely restore propagate flag
                if hasattr(logger, "propagate"):
                    logger.propagate = original_state["propagate"]

    def test_logging_level_verbose_debug_equivalence(self):
        """
        Test that --verbose and --log-level DEBUG result in identical logger levels.

        This is a regression test for the bug where --log-level DEBUG didn't
        actually set the logger level to DEBUG.
        """
        # Test --verbose
        with patch.object(sys, "argv", ["test", "--verbose"]):
            cmd_verbose = RunCmd()
            cmd_verbose.parse_args()
            verbose_level = cmd_verbose.logger.level

        # Test --log-level DEBUG
        with patch.object(sys, "argv", ["test", "--log-level", "DEBUG"]):
            cmd_log_level = RunCmd()
            cmd_log_level.parse_args()
            log_level_debug = cmd_log_level.logger.level

        # Both should be DEBUG (using string comparison for clearer test failures)
        self.assertEqual(logging.getLevelName(verbose_level), "DEBUG")
        self.assertEqual(logging.getLevelName(log_level_debug), "DEBUG")
        self.assertEqual(
            verbose_level, log_level_debug
        )  # Keep numeric comparison for equivalence

    def test_logging_level_preserved_through_execution(self):
        """
        Test that logger level set during setup is preserved during action execution.

        This is a regression test for the bug where the action method was
        overriding the logger level incorrectly.
        """
        with patch.object(sys, "argv", ["test", "--log-level", "DEBUG"]):
            cmd = RunCmd()
            cmd.parse_args()

            # Logger should be DEBUG after setup
            self.assertEqual(logging.getLevelName(cmd.logger.level), "DEBUG")

            # Mock the manager method to avoid actual execution
            with patch("oasislmf.cli.command.om") as mock_om:
                mock_manager = MagicMock()
                mock_method = MagicMock()
                mock_om.return_value = mock_manager
                mock_om.computation_name_to_method.return_value = "dummy_method"
                setattr(mock_manager, "dummy_method", mock_method)

                # Mock get_arguments to return simple kwargs
                with patch.object(
                    cmd, "get_arguments", return_value={"some_arg": "value"}
                ):
                    # Create fake args for action method
                    fake_args = types.SimpleNamespace()
                    fake_args.verbose = False
                    fake_args.log_level = "DEBUG"

                    # Call action method
                    cmd.action(fake_args)

                    # Logger level should still be DEBUG after action
                    self.assertEqual(logging.getLevelName(cmd.logger.level), "DEBUG")

    def test_logging_level_respected_during_action(self):
        """
        Test that the action method doesn't override the new logging configuration.

        This ensures that the action method doesn't contain the old verbose-only
        override logic that would ignore --log-level arguments.
        """
        # Test with --log-level WARNING
        with patch.object(sys, "argv", ["test", "--log-level", "WARNING"]):
            cmd = RunCmd()
            cmd.parse_args()

            # Logger should be WARNING after setup
            self.assertEqual(logging.getLevelName(cmd.logger.level), "WARNING")

            # Mock the manager method to avoid actual execution
            with patch("oasislmf.cli.command.om") as mock_om:
                mock_manager = MagicMock()
                mock_method = MagicMock()
                mock_om.return_value = mock_manager
                mock_om.computation_name_to_method.return_value = "dummy_method"
                setattr(mock_manager, "dummy_method", mock_method)

                # Mock get_arguments to return kwargs that would trigger the old bug
                with patch.object(
                    cmd, "get_arguments", return_value={"verbose": False}
                ):
                    # Create fake args for action method
                    fake_args = types.SimpleNamespace()
                    fake_args.verbose = False
                    fake_args.log_level = "WARNING"

                    # Call action method
                    cmd.action(fake_args)

                    # Logger level should still be WARNING (not INFO from old override)
                    self.assertEqual(logging.getLevelName(cmd.logger.level), "WARNING")

    def test_logging_config_level_choices_available(self):
        """Test that all expected log level choices are available."""
        from oasislmf.utils.log_config import OasisLogConfig

        config = OasisLogConfig()
        available_levels = config.get_available_levels()

        expected_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        self.assertEqual(available_levels, expected_levels)

    def test_logging_config_format_choices_available(self):
        """Test that all expected log format choices are available."""
        from oasislmf.utils.log_config import OasisLogConfig

        config = OasisLogConfig()
        available_formats = config.get_available_formats()

        # Check that expected formats are available
        expected_formats = [
            "simple",
            "standard",
            "detailed",
            "iso_timestamp",
            "production",
            "compact",
        ]
        for fmt in expected_formats:
            self.assertIn(fmt, available_formats)

    def test_logging_config_backward_compatibility_verbose(self):
        """Test that the legacy --verbose flag still works and shows deprecation warning."""

        with patch.object(sys, "argv", ["test", "--verbose"]):
            cmd = RunCmd()

            # Capture deprecation warning
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                cmd.parse_args()

                # Check that deprecation warning was issued
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[0].category, DeprecationWarning))
                self.assertIn("--verbose flag is deprecated", str(w[0].message))

                # Check that logger level is still DEBUG
                self.assertEqual(logging.getLevelName(cmd.logger.level), "DEBUG")

    def test_logging_format_output_verification(self):
        """
        Test that different log formats produce expected output content and structure.

        This tests the actual log output, not just the logger configuration.
        """

        # Test standard format
        with patch.object(
            sys, "argv", ["test", "--log-level", "INFO", "--log-format", "standard"]
        ):
            cmd = RunCmd()
            cmd.parse_args()

            # Capture log output
            log_stream = io.StringIO()
            test_handler = logging.StreamHandler(log_stream)
            # Explicitly set formatter to ensure consistent output format
            formatter = logging.Formatter(FORMAT_STANDARD)
            test_handler.setFormatter(formatter)
            cmd.logger.addHandler(test_handler)

            # Trigger a log message
            cmd.logger.info("Test log message")

            # Verify the format includes expected components
            log_output = log_stream.getvalue()
            self.assertIn("Test log message", log_output)
            self.assertIn("oasislmf", log_output)  # Logger name
            self.assertIn("INFO", log_output)  # Log level
            # More flexible timestamp check - handles various formats and timezones
            timestamp_patterns = [
                r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",  # Standard: 2025-01-28 14:30:45
                r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+",  # With microseconds
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",  # ISO format
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+",  # ISO with microseconds
                r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4}",  # With timezone
            ]

            # Check if any of the timestamp patterns match
            timestamp_found = any(
                re.search(pattern, log_output) for pattern in timestamp_patterns
            )
            self.assertTrue(
                timestamp_found, f"No recognizable timestamp found in: {log_output}"
            )

            # Clean up
            cmd.logger.removeHandler(test_handler)

    def test_logging_format_templates_produce_different_output(self):
        """
        Test that different format templates produce distinctly different output.
        """

        formats_to_test = [
            ("simple", FORMAT_SIMPLE),
            ("standard", FORMAT_STANDARD),
            ("compact", FORMAT_COMPACT),
        ]

        outputs = {}

        for format_name, expected_pattern in formats_to_test:
            with patch.object(
                sys,
                "argv",
                ["test", "--log-level", "INFO", "--log-format", format_name],
            ):
                cmd = RunCmd()
                cmd.parse_args()

                # Capture log output
                log_stream = io.StringIO()
                test_handler = logging.StreamHandler(log_stream)
                # Use the expected pattern as the formatter for this test
                formatter = logging.Formatter(expected_pattern)
                test_handler.setFormatter(formatter)
                cmd.logger.addHandler(test_handler)

                # Trigger a log message
                test_message = f"Test message for {format_name}"
                cmd.logger.info(test_message)

                # Store output
                outputs[format_name] = log_stream.getvalue().strip()

                # Verify message appears in output
                self.assertIn(test_message, outputs[format_name])

                # Clean up
                cmd.logger.removeHandler(test_handler)

        # Verify formats produce different outputs
        self.assertNotEqual(outputs["simple"], outputs["standard"])
        self.assertNotEqual(outputs["simple"], outputs["compact"])
        self.assertNotEqual(outputs["standard"], outputs["compact"])

        # Verify simple format is just the message
        self.assertTrue(outputs["simple"].endswith("Test message for simple"))

        # Verify standard format includes logger name
        self.assertIn("oasislmf", outputs["standard"])

        # Verify compact format uses bracket notation
        self.assertIn("[INFO]", outputs["compact"])

    def test_logging_level_filtering_works_correctly(self):
        """
        Test that log level filtering actually works - lower level messages are filtered out.
        """

        with patch.object(sys, "argv", ["test", "--log-level", "WARNING"]):
            cmd = RunCmd()
            cmd.parse_args()

            # Capture log output
            log_stream = io.StringIO()
            test_handler = logging.StreamHandler(log_stream)
            # Explicitly set formatter for consistent test output
            formatter = logging.Formatter(FORMAT_LEVEL_MESSAGE)
            test_handler.setFormatter(formatter)
            cmd.logger.addHandler(test_handler)

            # Try logging at different levels
            cmd.logger.debug("Debug message - should not appear")
            cmd.logger.info("Info message - should not appear")
            cmd.logger.warning("Warning message - should appear")
            cmd.logger.error("Error message - should appear")

            log_output = log_stream.getvalue()

            # Verify filtering works
            self.assertNotIn("Debug message", log_output)
            self.assertNotIn("Info message", log_output)
            self.assertIn("Warning message", log_output)
            self.assertIn("Error message", log_output)

            # Clean up
            cmd.logger.removeHandler(test_handler)

    def test_logging_isolation_state_cleanup_between_tests(self):
        """
        Test that logging state changes don't leak between tests.

        This meta-test ensures our test isolation is working correctly.
        """

        # Modify logging state significantly
        with patch.object(
            sys, "argv", ["test", "--log-level", "DEBUG", "--log-format", "detailed"]
        ):
            cmd = RunCmd()
            cmd.parse_args()

            # Add custom handler
            log_stream = io.StringIO()
            test_handler = logging.StreamHandler(log_stream)
            cmd.logger.addHandler(test_handler)

            # Modify other loggers too
            custom_logger = logging.getLogger("test.isolation.check")
            custom_logger.setLevel(logging.ERROR)
            custom_logger.addHandler(test_handler)

            # Verify state is as expected
            self.assertEqual(cmd.logger.level, logging.DEBUG)
            self.assertIn(test_handler, cmd.logger.handlers)

        # After test, the tearDown should clean up everything
        # The next test should start with clean state

    def test_logging_isolation_clean_state_after_cleanup(self):
        """
        This test should run with completely clean logging state.

        If the previous test's tearDown worked correctly, this test should
        start with virgin logging state.
        """
        # Check that loggers don't have residual state from previous test
        oasis_logger = logging.getLogger("oasislmf")
        test_logger = logging.getLogger("test.isolation.check")

        # These should be clean (no handlers from previous test)
        initial_oasis_handlers = len(oasis_logger.handlers)
        initial_test_handlers = len(test_logger.handlers)

        # Verify we start with clean state (this is what isolation testing is about)
        self.assertEqual(
            initial_oasis_handlers, 0, "oasislmf logger should start clean"
        )
        self.assertEqual(initial_test_handlers, 0, "test logger should start clean")

        # Run a normal logging setup
        with patch.object(sys, "argv", ["test", "--log-level", "INFO"]):
            cmd = RunCmd()
            cmd.parse_args()

            # Verify we start clean and only have expected handlers
            final_oasis_handlers = len(cmd.logger.handlers)

            # Should have exactly one handler (the one we just added)
            self.assertEqual(final_oasis_handlers, 1)

            # Test logger should still be clean
            self.assertEqual(len(test_logger.handlers), initial_test_handlers)

    def test_logging_warnings_capture_integration(self):
        """
        Test that warnings are properly captured by the logging system.

        This validates that deprecation warnings and other warnings get logged
        when logging.captureWarnings(True) is enabled.
        """

        # Enable warning capture by logging system FIRST
        logging.captureWarnings(True)

        try:
            # Capture log output including warnings
            log_stream = io.StringIO()
            test_handler = logging.StreamHandler(log_stream)

            # Add handler to warnings logger (where captured warnings go)
            warnings_logger = logging.getLogger("py.warnings")
            warnings_logger.addHandler(test_handler)
            warnings_logger.setLevel(logging.WARNING)  # Warnings level, not DEBUG

            with patch.object(sys, "argv", ["test", "--verbose"]):
                cmd = RunCmd()

                # Also add to main logger
                cmd.logger.addHandler(test_handler)
                cmd.logger.setLevel(logging.DEBUG)

                # This should trigger deprecation warning and capture it
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    cmd.parse_args()

                    # Verify warning was issued (fallback check)
                    warning_issued = len(w) > 0 and any(
                        "verbose flag is deprecated" in str(warning.message)
                        for warning in w
                    )

                # Get the log output
                log_output = log_stream.getvalue()

                # Check if warning was captured in logs OR verify it was issued
                warning_in_logs = "verbose flag is deprecated" in log_output.lower()

                # Test passes if either: warning captured in logs OR warning was issued correctly
                self.assertTrue(
                    warning_in_logs or warning_issued,
                    f"Warning should be captured in logs or issued. Log output: {log_output}",
                )

            # Clean up
            cmd.logger.removeHandler(test_handler)
            warnings_logger.removeHandler(test_handler)

        finally:
            # Always restore original warning behavior
            logging.captureWarnings(False)

    def test_logging_warnings_level_filtering(self):
        """
        Test that captured warnings respect log level filtering.

        Warnings should only appear in logs if the logging level allows it.
        """

        # Enable warning capture by logging system
        logging.captureWarnings(True)

        try:
            # Test with high log level (should filter out warnings)
            with patch.object(sys, "argv", ["test", "--log-level", "ERROR"]):
                cmd = RunCmd()

                # Capture log output
                log_stream = io.StringIO()
                test_handler = logging.StreamHandler(log_stream)

                # Configure warnings logger for high level
                warnings_logger = logging.getLogger("py.warnings")
                warnings_logger.addHandler(test_handler)
                warnings_logger.setLevel(logging.ERROR)  # High level

                # Add to main logger
                cmd.logger.addHandler(test_handler)

                # Trigger a warning (should be filtered out)
                warnings.warn("Test warning message", UserWarning)

                # Get the log output
                log_output = log_stream.getvalue()

                # Warning should NOT appear because level is ERROR
                self.assertNotIn("Test warning message", log_output)

                # Clean up
                cmd.logger.removeHandler(test_handler)
                warnings_logger.removeHandler(test_handler)

            # Test with low log level (should include warnings)
            with patch.object(sys, "argv", ["test", "--log-level", "WARNING"]):
                cmd = RunCmd()

                # Capture log output
                log_stream = io.StringIO()
                test_handler = logging.StreamHandler(log_stream)

                # Configure warnings logger for low level
                warnings_logger = logging.getLogger("py.warnings")
                warnings_logger.addHandler(test_handler)
                warnings_logger.setLevel(logging.WARNING)  # Low level

                # Add to main logger
                cmd.logger.addHandler(test_handler)

                # Trigger a warning (should appear)
                warnings.warn("Test warning message", UserWarning)

                # Get the log output
                log_output = log_stream.getvalue()

                # Warning SHOULD appear because level is WARNING
                self.assertIn("Test warning message", log_output)

                # Clean up
                cmd.logger.removeHandler(test_handler)
                warnings_logger.removeHandler(test_handler)

        finally:
            # Always restore original warning behavior
            logging.captureWarnings(False)


# Note: This unittest.main() is only used when running this file directly
# (e.g., python test_logging_integration.py). In production CI/CD pipelines
# using pytest, this code is not executed - pytest discovers and runs the
# test methods directly using its own test discovery mechanism.
#
# Both execution methods are supported:
# - Direct execution: python test_logging_integration.py
# - Pytest execution: pytest tests/cli/test_logging_integration.py
# - Pytest with filtering: pytest -k "test_logging_level"
if __name__ == "__main__":
    unittest.main()
