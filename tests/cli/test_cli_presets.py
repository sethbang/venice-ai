"""
Tests for preset management system
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from venice_ai.cli.presets import (
    apply_preset_to_config,
    delete_preset,
    get_builtin_presets,
    get_presets_dir,
    list_presets,
    load_preset,
    save_preset,
)


@pytest.fixture
def temp_presets_dir(tmp_path):
    """Fixture providing temporary presets directory"""
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)
    return presets_dir


@pytest.fixture
def sample_preset_config():
    """Fixture providing sample preset configuration"""
    return {
        "steps": 30,
        "cfg_scale": 7.5,
        "format": "png",
        "safe_mode": True,
        "embed_exif": True,
    }


@pytest.fixture
def mock_console_functions():
    """Mock console output functions to avoid cluttering test output"""
    with (
        patch("venice_ai.cli.presets.print_error") as mock_error,
        patch("venice_ai.cli.presets.print_success") as mock_success,
    ):
        yield {"error": mock_error, "success": mock_success}


class TestGetPresetsDir:
    """Test get_presets_dir function"""

    def test_creates_directory_if_not_exists(self, tmp_path):
        """Test that get_presets_dir creates directory if it doesn't exist"""
        test_dir = tmp_path / "test_presets"

        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", test_dir):
            result = get_presets_dir()

            assert result == test_dir
            assert test_dir.exists()
            assert test_dir.is_dir()

    def test_returns_existing_directory(self, temp_presets_dir):
        """Test that get_presets_dir returns existing directory"""
        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            result = get_presets_dir()

            assert result == temp_presets_dir
            assert result.exists()


class TestSavePreset:
    """Test save_preset function"""

    def test_save_preset_successfully(
        self, temp_presets_dir, sample_preset_config, mock_console_functions
    ):
        """Test successfully saving a preset"""
        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            result = save_preset("test-preset", sample_preset_config)

            assert result is True
            preset_file = temp_presets_dir / "test-preset.json"
            assert preset_file.exists()

            # Verify file content
            with open(preset_file) as f:
                saved_data = json.load(f)

            assert saved_data["name"] == "test-preset"
            assert saved_data["config"] == sample_preset_config
            assert "created_at" in saved_data
            assert "updated_at" in saved_data

            # Verify timestamps are valid ISO format
            datetime.fromisoformat(saved_data["created_at"])
            datetime.fromisoformat(saved_data["updated_at"])

            # Verify success message was printed
            mock_console_functions["success"].assert_called_once()

    def test_save_preset_creates_directory(
        self, tmp_path, sample_preset_config, mock_console_functions
    ):
        """Test that save_preset creates directory if it doesn't exist"""
        presets_dir = tmp_path / "new_presets"

        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", presets_dir):
            result = save_preset("test", sample_preset_config)

            assert result is True
            assert presets_dir.exists()
            assert (presets_dir / "test.json").exists()

    def test_save_preset_overwrites_existing(
        self, temp_presets_dir, sample_preset_config, mock_console_functions
    ):
        """Test that save_preset overwrites existing preset"""
        preset_name = "overwrite-test"

        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            # Save first preset
            save_preset(preset_name, {"steps": 10})

            # Save with new config
            result = save_preset(preset_name, sample_preset_config)

            assert result is True

            # Verify new content
            preset_file = temp_presets_dir / f"{preset_name}.json"
            with open(preset_file) as f:
                saved_data = json.load(f)

            assert saved_data["config"] == sample_preset_config
            assert saved_data["config"]["steps"] == 30

    def test_save_preset_handles_special_characters_in_name(
        self, temp_presets_dir, sample_preset_config, mock_console_functions
    ):
        """Test saving preset with special characters in name"""
        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            # Most filesystems support these characters
            result = save_preset("my-preset_v2", sample_preset_config)

            assert result is True
            assert (temp_presets_dir / "my-preset_v2.json").exists()

    def test_save_preset_handles_write_error(
        self, tmp_path, sample_preset_config, mock_console_functions
    ):
        """Test that save_preset handles write errors gracefully"""
        # Create a file instead of directory to cause error
        bad_path = tmp_path / "bad_presets"
        bad_path.write_text("not a directory")

        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", bad_path):
            result = save_preset("test", sample_preset_config)

            assert result is False
            mock_console_functions["error"].assert_called_once()


class TestLoadPreset:
    """Test load_preset function"""

    def test_load_existing_preset(
        self, temp_presets_dir, sample_preset_config, mock_console_functions
    ):
        """Test successfully loading an existing preset"""
        preset_name = "test-load"

        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            # Save preset first
            save_preset(preset_name, sample_preset_config)

            # Load it
            result = load_preset(preset_name)

            assert result == sample_preset_config

    def test_load_preset_updates_timestamp(
        self, temp_presets_dir, sample_preset_config, mock_console_functions
    ):
        """Test that load_preset updates the updated_at timestamp"""
        preset_name = "timestamp-test"

        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            # Save preset
            save_preset(preset_name, sample_preset_config)

            # Get original timestamp
            preset_file = temp_presets_dir / f"{preset_name}.json"
            with open(preset_file) as f:
                original_data = json.load(f)
            original_timestamp = original_data["updated_at"]

            # Load preset (this should update timestamp)
            load_preset(preset_name)

            # Check timestamp was updated
            with open(preset_file) as f:
                updated_data = json.load(f)

            assert updated_data["updated_at"] != original_timestamp

    def test_load_nonexistent_preset(self, temp_presets_dir, mock_console_functions):
        """Test loading a preset that doesn't exist returns None"""
        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            result = load_preset("nonexistent")

            assert result is None
            mock_console_functions["error"].assert_called_once()

    def test_load_preset_handles_corrupted_json(self, temp_presets_dir, mock_console_functions):
        """Test loading preset with corrupted JSON returns None"""
        preset_name = "corrupted"
        preset_file = temp_presets_dir / f"{preset_name}.json"

        # Write invalid JSON
        preset_file.write_text("{ invalid json content }")

        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            result = load_preset(preset_name)

            assert result is None
            mock_console_functions["error"].assert_called_once()

    def test_load_preset_handles_missing_config_key(self, temp_presets_dir, mock_console_functions):
        """Test loading preset with missing config key"""
        preset_name = "missing-config"
        preset_file = temp_presets_dir / f"{preset_name}.json"

        # Write JSON without config key
        preset_file.write_text(
            json.dumps({"name": preset_name, "created_at": datetime.now().isoformat()})
        )

        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            result = load_preset(preset_name)

            # Should return None if config is missing
            assert result is None


class TestListPresets:
    """Test list_presets function"""

    def test_list_empty_presets(self, temp_presets_dir, mock_console_functions):
        """Test listing presets when directory is empty"""
        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            result = list_presets()

            assert result == []

    def test_list_multiple_presets(
        self, temp_presets_dir, sample_preset_config, mock_console_functions
    ):
        """Test listing multiple presets"""
        presets = ["preset1", "preset2", "preset3"]

        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            # Save multiple presets
            for preset_name in presets:
                save_preset(preset_name, sample_preset_config)

            result = list_presets()

            assert len(result) == 3
            preset_names = [p["name"] for p in result]
            assert all(name in preset_names for name in presets)

    def test_list_presets_sorted_by_updated_at(
        self, temp_presets_dir, sample_preset_config, mock_console_functions
    ):
        """Test that presets are sorted by updated_at (newest first)"""
        import time

        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            # Save presets with small delays
            save_preset("first", sample_preset_config)
            time.sleep(0.01)
            save_preset("second", sample_preset_config)
            time.sleep(0.01)
            save_preset("third", sample_preset_config)

            result = list_presets()

            # Newest should be first
            assert result[0]["name"] == "third"
            assert result[1]["name"] == "second"
            assert result[2]["name"] == "first"

    def test_list_presets_includes_metadata(
        self, temp_presets_dir, sample_preset_config, mock_console_functions
    ):
        """Test that list_presets returns correct metadata structure"""
        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            save_preset("metadata-test", sample_preset_config)

            result = list_presets()

            assert len(result) == 1
            preset = result[0]

            assert "name" in preset
            assert "created_at" in preset
            assert "updated_at" in preset
            assert "config" in preset

            assert preset["name"] == "metadata-test"
            assert preset["config"] == sample_preset_config

    def test_list_presets_handles_non_json_files(
        self, temp_presets_dir, sample_preset_config, mock_console_functions
    ):
        """Test that list_presets ignores non-JSON files"""
        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            # Save a valid preset
            save_preset("valid", sample_preset_config)

            # Create non-JSON file
            (temp_presets_dir / "readme.txt").write_text("This is not a preset")

            result = list_presets()

            # Should only return the valid preset
            assert len(result) == 1
            assert result[0]["name"] == "valid"

    def test_list_presets_handles_corrupted_json(
        self, temp_presets_dir, sample_preset_config, mock_console_functions
    ):
        """Test that list_presets skips corrupted JSON files"""
        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            # Save a valid preset
            save_preset("valid", sample_preset_config)

            # Create corrupted JSON file
            (temp_presets_dir / "corrupted.json").write_text("{ invalid }")

            result = list_presets()

            # Should only return the valid preset, skipping corrupted
            assert len(result) == 1
            assert result[0]["name"] == "valid"

    def test_list_presets_handles_directory_error(self, tmp_path, mock_console_functions):
        """Test that list_presets handles directory access errors"""
        # Use a file path instead of directory to cause error
        bad_path = tmp_path / "not_a_directory.txt"
        bad_path.write_text("file")

        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", bad_path):
            result = list_presets()

            assert result == []
            mock_console_functions["error"].assert_called_once()


class TestDeletePreset:
    """Test delete_preset function"""

    def test_delete_existing_preset(
        self, temp_presets_dir, sample_preset_config, mock_console_functions
    ):
        """Test successfully deleting an existing preset"""
        preset_name = "to-delete"

        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            # Save preset first
            save_preset(preset_name, sample_preset_config)
            assert (temp_presets_dir / f"{preset_name}.json").exists()

            # Delete it
            result = delete_preset(preset_name)

            assert result is True
            assert not (temp_presets_dir / f"{preset_name}.json").exists()
            mock_console_functions["success"].assert_called()

    def test_delete_nonexistent_preset(self, temp_presets_dir, mock_console_functions):
        """Test deleting a preset that doesn't exist returns False"""
        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            result = delete_preset("nonexistent")

            assert result is False
            mock_console_functions["error"].assert_called_once()

    def test_delete_preset_handles_permission_error(
        self, temp_presets_dir, sample_preset_config, mock_console_functions
    ):
        """Test that delete_preset handles permission errors gracefully"""
        preset_name = "readonly"

        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            # Save preset
            save_preset(preset_name, sample_preset_config)

            # Mock unlink to raise PermissionError
            with patch.object(Path, "unlink", side_effect=PermissionError("No permission")):
                result = delete_preset(preset_name)

                assert result is False
                mock_console_functions["error"].assert_called()


class TestGetBuiltinPresets:
    """Test get_builtin_presets function"""

    def test_returns_dictionary(self):
        """Test that get_builtin_presets returns a dictionary"""
        result = get_builtin_presets()

        assert isinstance(result, dict)

    def test_returns_five_presets(self):
        """Test that get_builtin_presets returns 5 presets"""
        result = get_builtin_presets()

        assert len(result) == 5

    def test_has_required_presets(self):
        """Test that all required builtin presets exist"""
        result = get_builtin_presets()

        required_presets = [
            "photorealistic",
            "artistic",
            "quick",
            "high-quality",
            "creative",
        ]

        for preset_name in required_presets:
            assert preset_name in result

    def test_presets_have_required_keys(self):
        """Test that each preset has required configuration keys"""
        result = get_builtin_presets()

        required_keys = ["steps", "cfg_scale", "format", "safe_mode"]

        for preset_name, preset_config in result.items():
            for key in required_keys:
                assert key in preset_config, f"Preset '{preset_name}' missing key '{key}'"

    def test_photorealistic_preset_config(self):
        """Test photorealistic preset has expected values"""
        result = get_builtin_presets()

        preset = result["photorealistic"]
        assert preset["steps"] == 30
        assert preset["cfg_scale"] == 7.5
        assert preset["format"] == "png"
        assert preset["safe_mode"] is True
        assert preset["embed_exif"] is True

    def test_artistic_preset_config(self):
        """Test artistic preset has expected values"""
        result = get_builtin_presets()

        preset = result["artistic"]
        assert preset["steps"] == 25
        assert preset["cfg_scale"] == 9.0
        assert preset["format"] == "webp"
        assert preset["safe_mode"] is True

    def test_quick_preset_config(self):
        """Test quick preset has lower steps for faster generation"""
        result = get_builtin_presets()

        preset = result["quick"]
        assert preset["steps"] == 15
        assert preset["cfg_scale"] == 7.0
        assert preset["format"] == "webp"
        assert preset["safe_mode"] is True

    def test_high_quality_preset_config(self):
        """Test high-quality preset has higher steps"""
        result = get_builtin_presets()

        preset = result["high-quality"]
        assert preset["steps"] == 50
        assert preset["cfg_scale"] == 8.0
        assert preset["format"] == "png"
        assert preset["safe_mode"] is True

    def test_creative_preset_config(self):
        """Test creative preset has lower cfg_scale and safe_mode off"""
        result = get_builtin_presets()

        preset = result["creative"]
        assert preset["steps"] == 20
        assert preset["cfg_scale"] == 5.0
        assert preset["format"] == "webp"
        assert preset["safe_mode"] is False

    def test_all_presets_have_descriptions(self):
        """Test that all builtin presets have description field"""
        result = get_builtin_presets()

        for preset_name, preset_config in result.items():
            assert "description" in preset_config, f"Preset '{preset_name}' missing description"
            assert isinstance(preset_config["description"], str)
            assert len(preset_config["description"]) > 0


class TestApplyPresetToConfig:
    """Test apply_preset_to_config function"""

    def test_merge_preset_into_empty_config(self, sample_preset_config):
        """Test merging preset into empty config"""
        config = {}

        result = apply_preset_to_config(config, sample_preset_config)

        assert result == sample_preset_config
        # Verify original config wasn't modified
        assert config == {}

    def test_preset_doesnt_override_existing_values(self, sample_preset_config):
        """Test that preset doesn't override existing non-None values"""
        config = {
            "steps": 40,  # Different from preset
            "format": "jpg",  # Different from preset
        }

        result = apply_preset_to_config(config, sample_preset_config)

        # Existing values should be preserved
        assert result["steps"] == 40
        assert result["format"] == "jpg"

        # Preset values should be added for missing keys
        assert result["cfg_scale"] == sample_preset_config["cfg_scale"]
        assert result["safe_mode"] == sample_preset_config["safe_mode"]

    def test_preset_overrides_none_values(self, sample_preset_config):
        """Test that preset overrides None values in config"""
        config = {"steps": None, "cfg_scale": 8.0}

        result = apply_preset_to_config(config, sample_preset_config)

        # None value should be overridden
        assert result["steps"] == sample_preset_config["steps"]

        # Existing non-None value should be preserved
        assert result["cfg_scale"] == 8.0

    def test_empty_preset_returns_config_unchanged(self):
        """Test that empty preset doesn't modify config"""
        config = {"steps": 30, "format": "png"}
        empty_preset = {}

        result = apply_preset_to_config(config, empty_preset)

        assert result == config

    def test_doesnt_modify_original_config(self, sample_preset_config):
        """Test that original config dict is not modified"""
        original_config = {"steps": 40}
        config_copy = original_config.copy()

        result = apply_preset_to_config(original_config, sample_preset_config)

        # Original should be unchanged
        assert original_config == config_copy

        # Result should have merged values
        assert result != original_config
        assert "cfg_scale" in result

    def test_complex_merge_scenario(self):
        """Test complex merge with multiple override rules"""
        config = {
            "steps": 25,  # Has value - should keep
            "cfg_scale": None,  # None - should take preset value
            "format": "webp",  # Has value - should keep
            "embed_exif": None,  # None - should take preset value if in preset
        }

        preset = {"steps": 30, "cfg_scale": 7.5, "safe_mode": True, "embed_exif": True}

        result = apply_preset_to_config(config, preset)

        assert result["steps"] == 25  # Kept original
        assert result["cfg_scale"] == 7.5  # Took from preset (was None)
        assert result["format"] == "webp"  # Kept original
        assert result["safe_mode"] is True  # Added from preset
        assert result["embed_exif"] is True  # Took from preset (was None)


class TestPresetIntegration:
    """Integration tests for preset workflow"""

    def test_save_load_delete_workflow(
        self, temp_presets_dir, sample_preset_config, mock_console_functions
    ):
        """Test complete workflow: save, load, delete"""
        preset_name = "workflow-test"

        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            # Save
            save_result = save_preset(preset_name, sample_preset_config)
            assert save_result is True

            # Load
            loaded_config = load_preset(preset_name)
            assert loaded_config == sample_preset_config

            # Delete
            delete_result = delete_preset(preset_name)
            assert delete_result is True

            # Verify deleted
            load_result = load_preset(preset_name)
            assert load_result is None

    def test_list_after_multiple_operations(
        self, temp_presets_dir, sample_preset_config, mock_console_functions
    ):
        """Test listing after various operations"""
        with patch("venice_ai.cli.presets.DEFAULT_PRESETS_DIR", temp_presets_dir):
            # Save multiple presets
            save_preset("preset1", sample_preset_config)
            save_preset("preset2", {"steps": 20})
            save_preset("preset3", {"steps": 40})

            # List should show 3
            presets = list_presets()
            assert len(presets) == 3

            # Delete one
            delete_preset("preset2")

            # List should show 2
            presets = list_presets()
            assert len(presets) == 2
            preset_names = [p["name"] for p in presets]
            assert "preset1" in preset_names
            assert "preset3" in preset_names
            assert "preset2" not in preset_names

    def test_builtin_preset_can_be_applied_to_config(self):
        """Test that builtin presets work with apply_preset_to_config"""
        builtin_presets = get_builtin_presets()

        for _preset_name, preset_config in builtin_presets.items():
            config = {"steps": None, "format": None}

            result = apply_preset_to_config(config, preset_config)

            # Should have values from preset
            assert result["steps"] == preset_config["steps"]
            assert result["format"] == preset_config["format"]
            assert result["cfg_scale"] == preset_config["cfg_scale"]
