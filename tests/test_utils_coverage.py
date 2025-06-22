"""
Additional test cases for utils.py to improve coverage.
"""

import pytest
from unittest.mock import patch, MagicMock
import importlib.util
from venice_ai.utils import import_module_from_path


class TestUtilsCoverage:
    """Additional test cases to improve coverage of utils.py"""

    def test_import_module_from_path_spec_is_none(self):
        """Test that import_module_from_path handles spec being None."""
        with patch('importlib.util.spec_from_file_location', return_value=None):
            with pytest.raises(ImportError):
                import_module_from_path("non_existent_module", "non_existent_path")

    def test_import_module_from_path_module_is_none(self):
        """Test that import_module_from_path handles module being None."""
        # Create a real spec object to avoid mock issues
        spec = importlib.util.spec_from_loader("some_module", loader=MagicMock())
        assert spec is not None
        with patch('importlib.util.spec_from_file_location', return_value=spec):
            with patch('importlib.util.module_from_spec', return_value=None):
                with pytest.raises(ImportError):
                    import_module_from_path("some_module", "some_path")

    def test_import_module_from_path_no_loader(self):
        """Test that import_module_from_path handles spec.loader being None."""
        # Create a real spec object and then remove the loader
        spec = importlib.util.spec_from_loader("some_module", loader=MagicMock())
        assert spec is not None
        spec.loader = None
        with patch('importlib.util.spec_from_file_location', return_value=spec):
            # The function should not raise an error, but return a module
            module = import_module_from_path("some_module", "some_path")
            assert module is not None

    def test_import_module_from_path_loader_no_exec_module(self):
        """Test that import_module_from_path handles loader without exec_module."""
        # Create a real spec object with a loader that's missing exec_module
        loader = MagicMock()
        del loader.exec_module
        spec = importlib.util.spec_from_loader("some_module", loader=loader)
        assert spec is not None
        with patch('importlib.util.spec_from_file_location', return_value=spec):
            with pytest.raises(ImportError):
                import_module_from_path("some_module", "some_path")

    def test_import_module_from_path_exec_module_raises_importerror(self):
        """Test that import_module_from_path re-raises ImportError from exec_module."""
        # Create a real spec object with a loader that raises an error
        loader = MagicMock()
        loader.exec_module.side_effect = ImportError("Execution failed")
        spec = importlib.util.spec_from_loader("some_module", loader=loader)
        assert spec is not None
        with patch('importlib.util.spec_from_file_location', return_value=spec):
            with pytest.raises(ImportError, match="Execution failed"):
                import_module_from_path("some_module", "some_path")