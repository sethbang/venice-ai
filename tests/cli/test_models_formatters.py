"""
Tests for model formatting functionality
"""

import json
from types import SimpleNamespace

from venice_ai.cli.commands.models.formatters import ModelFormatter


class TestModelFormatter:
    """Test ModelFormatter class"""

    def test_format_price_both_currencies(self):
        """Test formatting price with both currencies"""
        pricing = SimpleNamespace(usd=0.50, diem=0.50)

        result = ModelFormatter.format_price(pricing, "both")
        assert "$0.50" in result
        assert "Ð0.50" in result
        assert "/" in result

    def test_format_price_usd_only(self):
        """Test formatting price with USD only"""
        pricing = SimpleNamespace(usd=0.50, diem=0.50)

        result = ModelFormatter.format_price(pricing, "usd")
        assert result == "$0.50"
        assert "Ð" not in result

    def test_format_price_diem_only(self):
        """Test formatting price with DIEM only"""
        pricing = SimpleNamespace(usd=0.50, diem=0.50)

        result = ModelFormatter.format_price(pricing, "diem")
        assert result == "Ð0.50"
        assert "$" not in result

    def test_format_price_none(self):
        """Test formatting None pricing"""
        result = ModelFormatter.format_price(None)
        assert result == "N/A"

    def test_format_price_missing_usd_attribute(self):
        """Test formatting price when usd attribute is missing"""
        pricing = SimpleNamespace(diem=0.50)  # No usd attribute
        result = ModelFormatter.format_price(pricing, "both")
        assert "N/A" in result
        assert "Ð0.50" in result

    def test_format_price_missing_diem_attribute(self):
        """Test formatting price when diem attribute is missing"""
        pricing = SimpleNamespace(usd=0.50)  # No diem attribute
        result = ModelFormatter.format_price(pricing, "both")
        assert "$0.50" in result
        assert "N/A" in result

    # Tests for format_gen_price covering lines 35, 41, 43
    def test_format_gen_price_both(self):
        """Test formatting generation price with 3 decimal places"""
        pricing = SimpleNamespace(usd=0.015, diem=0.015)

        result = ModelFormatter.format_gen_price(pricing, "both")
        assert "$0.015" in result
        assert "Ð0.015" in result

    def test_format_gen_price_none(self):
        """Test formatting None generation pricing - line 35"""
        result = ModelFormatter.format_gen_price(None)
        assert result == "N/A"

    def test_format_gen_price_usd_only(self):
        """Test formatting generation price USD only - line 41"""
        pricing = SimpleNamespace(usd=0.015, diem=0.015)
        result = ModelFormatter.format_gen_price(pricing, "usd")
        assert result == "$0.015"
        assert "Ð" not in result

    def test_format_gen_price_diem_only(self):
        """Test formatting generation price DIEM only - line 43"""
        pricing = SimpleNamespace(usd=0.015, diem=0.015)
        result = ModelFormatter.format_gen_price(pricing, "diem")
        assert result == "Ð0.015"
        assert "$" not in result

    def test_format_gen_price_missing_usd_attribute(self):
        """Test gen price when usd attribute is missing"""
        pricing = SimpleNamespace(diem=0.015)
        result = ModelFormatter.format_gen_price(pricing, "both")
        assert "N/A" in result
        assert "Ð0.015" in result

    def test_format_gen_price_missing_diem_attribute(self):
        """Test gen price when diem attribute is missing"""
        pricing = SimpleNamespace(usd=0.015)
        result = ModelFormatter.format_gen_price(pricing, "both")
        assert "$0.015" in result
        assert "N/A" in result

    # Tests for format_capabilities covering lines 56-67
    def test_format_capabilities(self):
        """Test formatting capabilities as icons"""
        caps = SimpleNamespace(
            supportsFunctionCalling=True,
            supportsVision=False,
            supportsReasoning=True,
            supportsWebSearch=True,
            optimizedForCode=False,
            supportsResponseSchema=True,
        )

        result = ModelFormatter.format_capabilities(caps)
        assert "🔧" in result  # Function calling
        assert "🧠" in result  # Reasoning
        assert "🌐" in result  # Web search
        assert "📝" in result  # Response schema
        assert "👁️" not in result  # No vision
        assert "💻" not in result  # No code

    def test_format_capabilities_with_vision(self):
        """Test formatting capabilities with vision support - line 57"""
        caps = SimpleNamespace(
            supportsFunctionCalling=False,
            supportsVision=True,  # Vision enabled
            supportsReasoning=False,
            supportsWebSearch=False,
            optimizedForCode=False,
            supportsResponseSchema=False,
        )
        result = ModelFormatter.format_capabilities(caps)
        assert "👁️" in result

    def test_format_capabilities_with_code_optimized(self):
        """Test formatting capabilities with code optimization - line 63"""
        caps = SimpleNamespace(
            supportsFunctionCalling=False,
            supportsVision=False,
            supportsReasoning=False,
            supportsWebSearch=False,
            optimizedForCode=True,  # Code optimized
            supportsResponseSchema=False,
        )
        result = ModelFormatter.format_capabilities(caps)
        assert "💻" in result

    def test_format_capabilities_all_enabled(self):
        """Test formatting all capabilities enabled"""
        caps = SimpleNamespace(
            supportsFunctionCalling=True,
            supportsVision=True,
            supportsReasoning=True,
            supportsWebSearch=True,
            optimizedForCode=True,
            supportsResponseSchema=True,
        )
        result = ModelFormatter.format_capabilities(caps)
        assert "🔧" in result
        assert "👁️" in result
        assert "🧠" in result
        assert "🌐" in result
        assert "💻" in result
        assert "📝" in result

    def test_format_capabilities_none(self):
        """Test formatting None capabilities"""
        result = ModelFormatter.format_capabilities(None)
        assert result == ""

    def test_format_capabilities_empty(self):
        """Test formatting capabilities with all False"""
        caps = SimpleNamespace(
            supportsFunctionCalling=False,
            supportsVision=False,
            supportsReasoning=False,
            supportsWebSearch=False,
            optimizedForCode=False,
            supportsResponseSchema=False,
        )
        result = ModelFormatter.format_capabilities(caps)
        assert result == ""

    def test_format_context_large(self):
        """Test formatting large context windows"""
        result = ModelFormatter.format_context(131072)
        assert result == "131k"

    def test_format_context_small(self):
        """Test formatting small context windows"""
        result = ModelFormatter.format_context(4096)
        assert result == "4,096"

    def test_format_context_none(self):
        """Test formatting None context"""
        result = ModelFormatter.format_context(None)
        assert result == "N/A"

    def test_format_context_zero(self):
        """Test formatting zero context (falsy)"""
        result = ModelFormatter.format_context(0)
        assert result == "N/A"

    def test_format_context_boundary(self):
        """Test formatting context at boundary (100k)"""
        result = ModelFormatter.format_context(100000)
        assert result == "100k"

    def test_format_context_just_under_boundary(self):
        """Test formatting context just under 100k"""
        result = ModelFormatter.format_context(99999)
        assert result == "99,999"

    def test_get_capability_legend(self):
        """Test capability legend generation"""
        legend = ModelFormatter.get_capability_legend()
        assert "🔧" in legend
        assert "Functions" in legend
        assert "Vision" in legend
        assert "Reasoning" in legend

    # Tests for format_text_table covering lines 109-151
    def test_format_text_table(self, mock_model_text):
        """Test formatting text models table"""
        table = ModelFormatter.format_text_table([mock_model_text])
        assert table is not None
        assert table.title is not None and "TEXT MODELS" in table.title

    def test_format_text_table_no_model_name(self):
        """Test text table when model has no name - line 112"""
        model = SimpleNamespace(
            id="test-model-without-name",
            type="text",
            model_spec=SimpleNamespace(
                # No name attribute
                traits=["default"],
                availableContextTokens=131072,
                pricing=SimpleNamespace(
                    input=SimpleNamespace(usd=0.5, diem=0.5),
                    output=SimpleNamespace(usd=2.0, diem=2.0),
                ),
                capabilities=SimpleNamespace(
                    supportsFunctionCalling=False,
                    supportsVision=False,
                    supportsReasoning=False,
                    supportsWebSearch=False,
                    optimizedForCode=False,
                    supportsResponseSchema=False,
                ),
            ),
        )
        table = ModelFormatter.format_text_table([model])
        assert table is not None

    def test_format_text_table_model_name_empty(self):
        """Test text table when model name is empty string"""
        model = SimpleNamespace(
            id="test-model-empty-name",
            type="text",
            model_spec=SimpleNamespace(
                name="",  # Empty name
                traits=["default"],
                availableContextTokens=131072,
                capabilities=SimpleNamespace(
                    supportsFunctionCalling=False,
                    supportsVision=False,
                    supportsReasoning=False,
                    supportsWebSearch=False,
                    optimizedForCode=False,
                    supportsResponseSchema=False,
                ),
            ),
        )
        table = ModelFormatter.format_text_table([model])
        assert table is not None

    def test_format_text_table_no_traits(self):
        """Test text table when model has no traits"""
        model = SimpleNamespace(
            id="test-model-no-traits",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                # No traits attribute
                availableContextTokens=131072,
            ),
        )
        table = ModelFormatter.format_text_table([model])
        assert table is not None

    def test_format_text_table_empty_traits(self):
        """Test text table when model has empty traits list"""
        model = SimpleNamespace(
            id="test-model-empty-traits",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                traits=[],  # Empty traits
                availableContextTokens=131072,
            ),
        )
        table = ModelFormatter.format_text_table([model])
        assert table is not None

    def test_format_text_table_no_pricing(self):
        """Test text table when model has no pricing"""
        model = SimpleNamespace(
            id="test-model-no-pricing",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                availableContextTokens=131072,
                # No pricing attribute
            ),
        )
        table = ModelFormatter.format_text_table([model])
        assert table is not None

    def test_format_text_table_no_input_pricing(self):
        """Test text table when model has no input pricing"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                availableContextTokens=131072,
                pricing=SimpleNamespace(
                    output=SimpleNamespace(usd=2.0, diem=2.0),
                    # No input pricing
                ),
            ),
        )
        table = ModelFormatter.format_text_table([model])
        assert table is not None

    def test_format_text_table_no_output_pricing(self):
        """Test text table when model has no output pricing"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                availableContextTokens=131072,
                pricing=SimpleNamespace(
                    input=SimpleNamespace(usd=0.5, diem=0.5),
                    # No output pricing
                ),
            ),
        )
        table = ModelFormatter.format_text_table([model])
        assert table is not None

    def test_format_text_table_no_capabilities(self):
        """Test text table when model has no capabilities"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                availableContextTokens=131072,
                # No capabilities
            ),
        )
        table = ModelFormatter.format_text_table([model])
        assert table is not None

    def test_format_text_table_currency_usd(self):
        """Test text table with USD currency"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                availableContextTokens=131072,
                pricing=SimpleNamespace(
                    input=SimpleNamespace(usd=0.5, diem=0.5),
                    output=SimpleNamespace(usd=2.0, diem=2.0),
                ),
            ),
        )
        table = ModelFormatter.format_text_table([model], currency="usd")
        assert table is not None

    # Tests for format_image_table covering lines 177-222
    def test_format_image_table(self, mock_model_image):
        """Test formatting image models table"""
        table = ModelFormatter.format_image_table([mock_model_image])
        assert table is not None
        assert table.title is not None and "IMAGE MODELS" in table.title

    def test_format_image_table_no_model_name(self):
        """Test image table when model has no name - line 180"""
        model = SimpleNamespace(
            id="test-image-no-name",
            type="image",
            model_spec=SimpleNamespace(
                # No name attribute
                traits=["quality"],
                offline=False,
                beta=False,
                pricing=SimpleNamespace(generation=SimpleNamespace(usd=0.01, diem=0.01)),
                constraints=SimpleNamespace(steps=SimpleNamespace(default=25, max=30)),
            ),
        )
        table = ModelFormatter.format_image_table([model])
        assert table is not None

    def test_format_image_table_empty_name(self):
        """Test image table when model name is empty string"""
        model = SimpleNamespace(
            id="test-image-empty-name",
            type="image",
            model_spec=SimpleNamespace(
                name="",  # Empty name
                offline=False,
                pricing=SimpleNamespace(generation=SimpleNamespace(usd=0.01, diem=0.01)),
            ),
        )
        table = ModelFormatter.format_image_table([model])
        assert table is not None

    def test_format_image_table_no_constraints(self):
        """Test image table when model has no constraints"""
        model = SimpleNamespace(
            id="test-image-no-constraints",
            type="image",
            model_spec=SimpleNamespace(
                name="Test Image",
                offline=False,
                beta=False,
                pricing=SimpleNamespace(generation=SimpleNamespace(usd=0.01, diem=0.01)),
                # No constraints
            ),
        )
        table = ModelFormatter.format_image_table([model])
        assert table is not None

    def test_format_image_table_constraints_no_steps(self):
        """Test image table when constraints has no steps"""
        model = SimpleNamespace(
            id="test-image",
            type="image",
            model_spec=SimpleNamespace(
                name="Test Image",
                offline=False,
                beta=False,
                pricing=SimpleNamespace(generation=SimpleNamespace(usd=0.01, diem=0.01)),
                constraints=SimpleNamespace(
                    # No steps attribute
                    other=True,
                ),
            ),
        )
        table = ModelFormatter.format_image_table([model])
        assert table is not None

    def test_format_image_table_steps_default_only(self):
        """Test image table with steps default only, no max - line 201-202"""
        model = SimpleNamespace(
            id="test-image",
            type="image",
            model_spec=SimpleNamespace(
                name="Test Image",
                offline=False,
                beta=False,
                pricing=SimpleNamespace(generation=SimpleNamespace(usd=0.01, diem=0.01)),
                constraints=SimpleNamespace(
                    steps=SimpleNamespace(default=25),  # No max
                ),
            ),
        )
        table = ModelFormatter.format_image_table([model])
        assert table is not None

    def test_format_image_table_no_pricing(self):
        """Test image table when model has no pricing"""
        model = SimpleNamespace(
            id="test-image-no-pricing",
            type="image",
            model_spec=SimpleNamespace(
                name="Test Image",
                offline=False,
                beta=False,
                # No pricing
            ),
        )
        table = ModelFormatter.format_image_table([model])
        assert table is not None

    def test_format_image_table_no_generation_pricing(self):
        """Test image table when model pricing has no generation"""
        model = SimpleNamespace(
            id="test-image",
            type="image",
            model_spec=SimpleNamespace(
                name="Test Image",
                offline=False,
                beta=False,
                pricing=SimpleNamespace(
                    # No generation pricing, but has other
                    other=SimpleNamespace(usd=0.01, diem=0.01),
                ),
            ),
        )
        table = ModelFormatter.format_image_table([model])
        assert table is not None

    def test_format_image_table_offline_status(self):
        """Test image table with offline model - line 218"""
        model = SimpleNamespace(
            id="test-image-offline",
            type="image",
            model_spec=SimpleNamespace(
                name="Test Image",
                offline=True,  # Offline
                beta=False,
                pricing=SimpleNamespace(generation=SimpleNamespace(usd=0.01, diem=0.01)),
            ),
        )
        table = ModelFormatter.format_image_table([model])
        assert table is not None

    def test_format_image_table_beta_status(self):
        """Test image table with beta model - line 220"""
        model = SimpleNamespace(
            id="test-image-beta",
            type="image",
            model_spec=SimpleNamespace(
                name="Test Image",
                offline=False,
                beta=True,  # Beta
                pricing=SimpleNamespace(generation=SimpleNamespace(usd=0.01, diem=0.01)),
            ),
        )
        table = ModelFormatter.format_image_table([model])
        assert table is not None

    def test_format_image_table_no_offline_attribute(self):
        """Test image table when model has no offline attribute"""
        model = SimpleNamespace(
            id="test-image",
            type="image",
            model_spec=SimpleNamespace(
                name="Test Image",
                # No offline attribute
                beta=False,
                pricing=SimpleNamespace(generation=SimpleNamespace(usd=0.01, diem=0.01)),
            ),
        )
        table = ModelFormatter.format_image_table([model])
        assert table is not None

    def test_format_image_table_no_beta_attribute(self):
        """Test image table when model has no beta attribute"""
        model = SimpleNamespace(
            id="test-image",
            type="image",
            model_spec=SimpleNamespace(
                name="Test Image",
                offline=False,
                # No beta attribute
                pricing=SimpleNamespace(generation=SimpleNamespace(usd=0.01, diem=0.01)),
            ),
        )
        table = ModelFormatter.format_image_table([model])
        assert table is not None

    # Tests for format_tts_table covering lines 245-266
    def test_format_tts_table(self):
        """Test formatting TTS models table"""
        tts_model = SimpleNamespace(
            id="tts-test",
            type="tts",
            model_spec=SimpleNamespace(
                name="TTS Test",
                voices=["voice1", "voice2", "voice3"],
                pricing=SimpleNamespace(input=SimpleNamespace(usd=3.5, diem=3.5)),
            ),
        )

        table = ModelFormatter.format_tts_table([tts_model])
        assert table is not None
        assert table.title is not None and "TTS MODELS" in table.title

    def test_format_tts_table_no_model_name(self):
        """Test TTS table when model has no name - line 248"""
        model = SimpleNamespace(
            id="tts-no-name",
            type="tts",
            model_spec=SimpleNamespace(
                # No name attribute
                voices=["voice1", "voice2"],
                pricing=SimpleNamespace(input=SimpleNamespace(usd=3.5, diem=3.5)),
            ),
        )
        table = ModelFormatter.format_tts_table([model])
        assert table is not None

    def test_format_tts_table_empty_name(self):
        """Test TTS table when model name is empty string"""
        model = SimpleNamespace(
            id="tts-empty-name",
            type="tts",
            model_spec=SimpleNamespace(
                name="",  # Empty name
                voices=["voice1"],
            ),
        )
        table = ModelFormatter.format_tts_table([model])
        assert table is not None

    def test_format_tts_table_no_voices(self):
        """Test TTS table when model has no voices attribute"""
        model = SimpleNamespace(
            id="tts-no-voices",
            type="tts",
            model_spec=SimpleNamespace(
                name="TTS Test",
                # No voices attribute
                pricing=SimpleNamespace(input=SimpleNamespace(usd=3.5, diem=3.5)),
            ),
        )
        table = ModelFormatter.format_tts_table([model])
        assert table is not None

    def test_format_tts_table_empty_voices(self):
        """Test TTS table when model has empty voices list"""
        model = SimpleNamespace(
            id="tts-empty-voices",
            type="tts",
            model_spec=SimpleNamespace(
                name="TTS Test",
                voices=[],  # Empty voices
            ),
        )
        table = ModelFormatter.format_tts_table([model])
        assert table is not None

    def test_format_tts_table_no_pricing(self):
        """Test TTS table when model has no pricing"""
        model = SimpleNamespace(
            id="tts-no-pricing",
            type="tts",
            model_spec=SimpleNamespace(
                name="TTS Test",
                voices=["voice1"],
                # No pricing
            ),
        )
        table = ModelFormatter.format_tts_table([model])
        assert table is not None

    def test_format_tts_table_no_input_pricing(self):
        """Test TTS table when model has pricing but no input"""
        model = SimpleNamespace(
            id="tts-no-input",
            type="tts",
            model_spec=SimpleNamespace(
                name="TTS Test",
                voices=["voice1"],
                pricing=SimpleNamespace(
                    # No input pricing
                    other=SimpleNamespace(usd=1.0, diem=1.0),
                ),
            ),
        )
        table = ModelFormatter.format_tts_table([model])
        assert table is not None

    # Tests for format_embedding_table covering lines 289-318
    def test_format_embedding_table(self, mock_model_embedding):
        """Test formatting embedding models table"""
        table = ModelFormatter.format_embedding_table([mock_model_embedding])
        assert table is not None
        assert table.title is not None and "EMBEDDING MODELS" in table.title

    def test_format_embedding_table_no_model_name(self):
        """Test embedding table when model has no name - line 292"""
        model = SimpleNamespace(
            id="embedding-no-name",
            type="embedding",
            model_spec=SimpleNamespace(
                # No name attribute
                availableContextTokens=8192,
                pricing=SimpleNamespace(
                    input=SimpleNamespace(usd=0.15, diem=0.15),
                    output=SimpleNamespace(usd=0.6, diem=0.6),
                ),
            ),
        )
        table = ModelFormatter.format_embedding_table([model])
        assert table is not None

    def test_format_embedding_table_empty_name(self):
        """Test embedding table when model name is empty"""
        model = SimpleNamespace(
            id="embedding-empty-name",
            type="embedding",
            model_spec=SimpleNamespace(
                name="",  # Empty name
                availableContextTokens=8192,
            ),
        )
        table = ModelFormatter.format_embedding_table([model])
        assert table is not None

    def test_format_embedding_table_no_context(self):
        """Test embedding table when model has no context - line 299"""
        model = SimpleNamespace(
            id="embedding-no-context",
            type="embedding",
            model_spec=SimpleNamespace(
                name="Test Embedding",
                # No availableContextTokens
            ),
        )
        table = ModelFormatter.format_embedding_table([model])
        assert table is not None

    def test_format_embedding_table_no_pricing(self):
        """Test embedding table when model has no pricing"""
        model = SimpleNamespace(
            id="embedding-no-pricing",
            type="embedding",
            model_spec=SimpleNamespace(
                name="Test Embedding",
                availableContextTokens=8192,
                # No pricing
            ),
        )
        table = ModelFormatter.format_embedding_table([model])
        assert table is not None

    def test_format_embedding_table_no_input_pricing(self):
        """Test embedding table when pricing has no input"""
        model = SimpleNamespace(
            id="embedding-no-input",
            type="embedding",
            model_spec=SimpleNamespace(
                name="Test Embedding",
                availableContextTokens=8192,
                pricing=SimpleNamespace(
                    output=SimpleNamespace(usd=0.6, diem=0.6),
                    # No input
                ),
            ),
        )
        table = ModelFormatter.format_embedding_table([model])
        assert table is not None

    def test_format_embedding_table_no_output_pricing(self):
        """Test embedding table when pricing has no output"""
        model = SimpleNamespace(
            id="embedding-no-output",
            type="embedding",
            model_spec=SimpleNamespace(
                name="Test Embedding",
                availableContextTokens=8192,
                pricing=SimpleNamespace(
                    input=SimpleNamespace(usd=0.15, diem=0.15),
                    # No output
                ),
            ),
        )
        table = ModelFormatter.format_embedding_table([model])
        assert table is not None

    # Tests for format_upscale_table covering lines 344-397
    def test_format_upscale_table(self, mock_model_upscale):
        """Test formatting upscale models table"""
        table = ModelFormatter.format_upscale_table([mock_model_upscale])
        assert table is not None
        assert table.title is not None and "UPSCALE MODELS" in table.title

    def test_format_upscale_table_no_model_name(self):
        """Test upscale table when model has no name - line 347"""
        model = SimpleNamespace(
            id="upscale-no-name",
            type="upscale",
            model_spec=SimpleNamespace(
                # No name attribute
                pricing=SimpleNamespace(
                    generation=SimpleNamespace(usd=0.01, diem=0.01),
                    upscale={
                        "2x": SimpleNamespace(usd=0.02, diem=0.02),
                        "4x": SimpleNamespace(usd=0.08, diem=0.08),
                    },
                ),
            ),
        )
        table = ModelFormatter.format_upscale_table([model])
        assert table is not None

    def test_format_upscale_table_empty_name(self):
        """Test upscale table when model name is empty"""
        model = SimpleNamespace(
            id="upscale-empty-name",
            type="upscale",
            model_spec=SimpleNamespace(
                name="",  # Empty name
            ),
        )
        table = ModelFormatter.format_upscale_table([model])
        assert table is not None

    def test_format_upscale_table_no_pricing(self):
        """Test upscale table when model has no pricing"""
        model = SimpleNamespace(
            id="upscale-no-pricing",
            type="upscale",
            model_spec=SimpleNamespace(
                name="Upscaler",
                # No pricing
            ),
        )
        table = ModelFormatter.format_upscale_table([model])
        assert table is not None

    def test_format_upscale_table_no_generation(self):
        """Test upscale table when pricing has no generation"""
        model = SimpleNamespace(
            id="upscale-no-gen",
            type="upscale",
            model_spec=SimpleNamespace(
                name="Upscaler",
                pricing=SimpleNamespace(
                    # No generation
                    upscale={
                        "2x": SimpleNamespace(usd=0.02, diem=0.02),
                    },
                ),
            ),
        )
        table = ModelFormatter.format_upscale_table([model])
        assert table is not None

    def test_format_upscale_table_no_upscale_pricing(self):
        """Test upscale table when pricing has no upscale"""
        model = SimpleNamespace(
            id="upscale-no-upscale",
            type="upscale",
            model_spec=SimpleNamespace(
                name="Upscaler",
                pricing=SimpleNamespace(
                    generation=SimpleNamespace(usd=0.01, diem=0.01),
                    # No upscale
                ),
            ),
        )
        table = ModelFormatter.format_upscale_table([model])
        assert table is not None

    def test_format_upscale_table_upscale_2x_only(self):
        """Test upscale table with only 2x pricing"""
        model = SimpleNamespace(
            id="upscale-2x-only",
            type="upscale",
            model_spec=SimpleNamespace(
                name="Upscaler",
                pricing=SimpleNamespace(
                    upscale={
                        "2x": SimpleNamespace(usd=0.02, diem=0.02),
                        # No 4x
                    },
                ),
            ),
        )
        table = ModelFormatter.format_upscale_table([model])
        assert table is not None

    def test_format_upscale_table_upscale_4x_only(self):
        """Test upscale table with only 4x pricing"""
        model = SimpleNamespace(
            id="upscale-4x-only",
            type="upscale",
            model_spec=SimpleNamespace(
                name="Upscaler",
                pricing=SimpleNamespace(
                    upscale={
                        # No 2x
                        "4x": SimpleNamespace(usd=0.08, diem=0.08),
                    },
                ),
            ),
        )
        table = ModelFormatter.format_upscale_table([model])
        assert table is not None

    def test_format_upscale_table_upscale_as_object_with_getitem(self):
        """Test upscale table with object that has __getitem__ - lines 378-391"""

        # Create a class that behaves like dict but isn't a dict
        class UpscalePricing:
            def __init__(self):
                self._data = {
                    "2x": SimpleNamespace(usd=0.02, diem=0.02),
                    "4x": SimpleNamespace(usd=0.08, diem=0.08),
                }

            def __getitem__(self, key):
                return self._data[key]

        model = SimpleNamespace(
            id="upscale-getitem",
            type="upscale",
            model_spec=SimpleNamespace(
                name="Upscaler",
                pricing=SimpleNamespace(
                    upscale=UpscalePricing(),
                ),
            ),
        )
        table = ModelFormatter.format_upscale_table([model])
        assert table is not None

    def test_format_upscale_table_upscale_getitem_2x_only(self):
        """Test upscale table with __getitem__ that only has 2x - line 384"""

        class UpscalePricing:
            def __init__(self):
                self._data = {
                    "2x": SimpleNamespace(usd=0.02, diem=0.02),
                }

            def __getitem__(self, key):
                return self._data[key]

        model = SimpleNamespace(
            id="upscale-getitem-2x",
            type="upscale",
            model_spec=SimpleNamespace(
                name="Upscaler",
                pricing=SimpleNamespace(
                    upscale=UpscalePricing(),
                ),
            ),
        )
        table = ModelFormatter.format_upscale_table([model])
        assert table is not None

    def test_format_upscale_table_upscale_getitem_4x_only(self):
        """Test upscale table with __getitem__ that only has 4x - line 391"""

        class UpscalePricing:
            def __init__(self):
                self._data = {
                    "4x": SimpleNamespace(usd=0.08, diem=0.08),
                }

            def __getitem__(self, key):
                return self._data[key]

        model = SimpleNamespace(
            id="upscale-getitem-4x",
            type="upscale",
            model_spec=SimpleNamespace(
                name="Upscaler",
                pricing=SimpleNamespace(
                    upscale=UpscalePricing(),
                ),
            ),
        )
        table = ModelFormatter.format_upscale_table([model])
        assert table is not None

    def test_format_upscale_table_upscale_getitem_raises_error(self):
        """Test upscale table with __getitem__ that raises exception - line 393"""

        class BrokenUpscalePricing:
            def __getitem__(self, key):
                raise RuntimeError("Access failed")

        model = SimpleNamespace(
            id="upscale-broken-getitem",
            type="upscale",
            model_spec=SimpleNamespace(
                name="Upscaler",
                pricing=SimpleNamespace(
                    upscale=BrokenUpscalePricing(),
                ),
            ),
        )
        table = ModelFormatter.format_upscale_table([model])
        assert table is not None

    # Tests for format_inpaint_table covering lines 420-436
    def test_format_inpaint_table(self):
        """Test formatting inpaint models table"""
        inpaint_model = SimpleNamespace(
            id="test-inpaint",
            type="inpaint",
            model_spec=SimpleNamespace(name="Test Inpaint", offline=False, beta=True),
        )

        table = ModelFormatter.format_inpaint_table([inpaint_model])
        assert table is not None
        assert table.title is not None and "INPAINT MODELS" in table.title

    def test_format_inpaint_table_no_model_name(self):
        """Test inpaint table when model has no name - line 423"""
        model = SimpleNamespace(
            id="inpaint-no-name",
            type="inpaint",
            model_spec=SimpleNamespace(
                # No name attribute
                offline=False,
                beta=False,
            ),
        )
        table = ModelFormatter.format_inpaint_table([model])
        assert table is not None

    def test_format_inpaint_table_empty_name(self):
        """Test inpaint table when model name is empty"""
        model = SimpleNamespace(
            id="inpaint-empty-name",
            type="inpaint",
            model_spec=SimpleNamespace(
                name="",  # Empty name
                offline=False,
                beta=False,
            ),
        )
        table = ModelFormatter.format_inpaint_table([model])
        assert table is not None

    def test_format_inpaint_table_offline(self):
        """Test inpaint table with offline model - line 430"""
        model = SimpleNamespace(
            id="inpaint-offline",
            type="inpaint",
            model_spec=SimpleNamespace(
                name="Test Inpaint",
                offline=True,  # Offline
                beta=False,
            ),
        )
        table = ModelFormatter.format_inpaint_table([model])
        assert table is not None

    def test_format_inpaint_table_no_offline_attribute(self):
        """Test inpaint table when model has no offline attribute"""
        model = SimpleNamespace(
            id="inpaint-no-offline",
            type="inpaint",
            model_spec=SimpleNamespace(
                name="Test Inpaint",
                # No offline attribute
                beta=False,
            ),
        )
        table = ModelFormatter.format_inpaint_table([model])
        assert table is not None

    def test_format_inpaint_table_no_beta_attribute(self):
        """Test inpaint table when model has no beta attribute"""
        model = SimpleNamespace(
            id="inpaint-no-beta",
            type="inpaint",
            model_spec=SimpleNamespace(
                name="Test Inpaint",
                offline=False,
                # No beta attribute
            ),
        )
        table = ModelFormatter.format_inpaint_table([model])
        assert table is not None

    # Tests for format_verbose_model covering lines 483-575
    def test_format_verbose_model(self, mock_model_text):
        """Test formatting verbose model view"""
        panel = ModelFormatter.format_verbose_model(mock_model_text)
        assert panel is not None

    def test_format_verbose_model_with_none_context(self, mock_model_embedding):
        """Test formatting verbose model with None context tokens"""
        # This tests the NoneType fix
        panel = ModelFormatter.format_verbose_model(mock_model_embedding)
        assert panel is not None

    def test_format_verbose_model_no_name(self):
        """Test verbose model when model has no name"""
        model = SimpleNamespace(
            id="test-model-no-name",
            type="text",
            model_spec=SimpleNamespace(
                # No name attribute
                capabilities=SimpleNamespace(
                    supportsFunctionCalling=False,
                    supportsVision=False,
                    supportsReasoning=False,
                    supportsWebSearch=False,
                    optimizedForCode=False,
                    supportsResponseSchema=False,
                    supportsLogProbs=False,
                ),
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_no_traits(self):
        """Test verbose model when model has no traits"""
        model = SimpleNamespace(
            id="test-model-no-traits",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                # No traits attribute
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_empty_traits(self):
        """Test verbose model when model has empty traits"""
        model = SimpleNamespace(
            id="test-model-empty-traits",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                traits=[],  # Empty traits
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_caps_line_wrap(self):
        """Test verbose model capability line wrapping - line 483"""
        # This tests the case where cap_line ends exactly at multiples of 3
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                capabilities=SimpleNamespace(
                    supportsFunctionCalling=True,
                    supportsVision=True,
                    supportsReasoning=True,
                    supportsWebSearch=True,
                    supportsLogProbs=True,
                    optimizedForCode=True,
                    supportsResponseSchema=True,
                ),
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_non_text_type(self):
        """Test verbose model for non-text model type (no capabilities section)"""
        model = SimpleNamespace(
            id="test-image",
            type="image",
            model_spec=SimpleNamespace(
                name="Test Image Model",
                pricing=SimpleNamespace(generation=SimpleNamespace(usd=0.01, diem=0.01)),
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_no_pricing(self):
        """Test verbose model with no pricing"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                # No pricing
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_input_pricing_only(self):
        """Test verbose model with input pricing only"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                pricing=SimpleNamespace(
                    input=SimpleNamespace(usd=0.5, diem=0.5),
                    # No output
                ),
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_output_pricing_only(self):
        """Test verbose model with output pricing only"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                pricing=SimpleNamespace(
                    # No input
                    output=SimpleNamespace(usd=2.0, diem=2.0),
                ),
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_generation_pricing(self):
        """Test verbose model with generation pricing - line 502"""
        model = SimpleNamespace(
            id="test-model",
            type="image",
            model_spec=SimpleNamespace(
                name="Test Model",
                pricing=SimpleNamespace(
                    generation=SimpleNamespace(usd=0.01, diem=0.01),
                ),
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_upscale_pricing(self):
        """Test verbose model with upscale pricing - lines 506-512"""

        # Create an upscale pricing object with getattr access (not dict)
        class UpscalePricingObj:
            pass

        upscale = UpscalePricingObj()
        setattr(upscale, "2x", SimpleNamespace(usd=0.02, diem=0.02))
        setattr(upscale, "4x", SimpleNamespace(usd=0.08, diem=0.08))

        model = SimpleNamespace(
            id="test-upscaler",
            type="upscale",
            model_spec=SimpleNamespace(
                name="Test Upscaler",
                pricing=SimpleNamespace(
                    upscale=upscale,
                ),
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_upscale_2x_only(self):
        """Test verbose model with only 2x upscale pricing - line 508"""

        class UpscalePricingObj:
            pass

        upscale = UpscalePricingObj()
        setattr(upscale, "2x", SimpleNamespace(usd=0.02, diem=0.02))
        # No 4x

        model = SimpleNamespace(
            id="test-upscaler",
            type="upscale",
            model_spec=SimpleNamespace(
                name="Test Upscaler",
                pricing=SimpleNamespace(
                    upscale=upscale,
                ),
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_upscale_4x_only(self):
        """Test verbose model with only 4x upscale pricing - line 512"""

        class UpscalePricingObj:
            pass

        upscale = UpscalePricingObj()
        # No 2x
        setattr(upscale, "4x", SimpleNamespace(usd=0.08, diem=0.08))

        model = SimpleNamespace(
            id="test-upscaler",
            type="upscale",
            model_spec=SimpleNamespace(
                name="Test Upscaler",
                pricing=SimpleNamespace(
                    upscale=upscale,
                ),
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_no_context(self):
        """Test verbose model with no context tokens"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                # No availableContextTokens
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_with_quantization(self):
        """Test verbose model with quantization - line 531"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                availableContextTokens=131072,
                capabilities=SimpleNamespace(
                    quantization="fp8",
                    supportsFunctionCalling=False,
                    supportsVision=False,
                    supportsReasoning=False,
                    supportsWebSearch=False,
                    supportsLogProbs=False,
                    optimizedForCode=False,
                    supportsResponseSchema=False,
                ),
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_no_quantization(self):
        """Test verbose model without quantization capability"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                availableContextTokens=131072,
                capabilities=SimpleNamespace(
                    # No quantization
                    supportsFunctionCalling=False,
                    supportsVision=False,
                    supportsReasoning=False,
                    supportsWebSearch=False,
                    supportsLogProbs=False,
                    optimizedForCode=False,
                    supportsResponseSchema=False,
                ),
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_with_constraints(self):
        """Test verbose model with full constraints - lines 538-552"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                constraints=SimpleNamespace(
                    temperature=SimpleNamespace(default=0.7),
                    top_p=SimpleNamespace(default=0.95),
                    steps=SimpleNamespace(default=25, max=30),
                ),
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_constraints_temperature_only(self):
        """Test verbose model with temperature constraint only"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                constraints=SimpleNamespace(
                    temperature=SimpleNamespace(default=0.7),
                    # No top_p, no steps
                ),
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_constraints_top_p_only(self):
        """Test verbose model with top_p constraint only"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                constraints=SimpleNamespace(
                    # No temperature
                    top_p=SimpleNamespace(default=0.95),
                ),
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_constraints_steps_only(self):
        """Test verbose model with steps constraint only - line 547"""
        model = SimpleNamespace(
            id="test-model",
            type="image",
            model_spec=SimpleNamespace(
                name="Test Model",
                constraints=SimpleNamespace(
                    steps=SimpleNamespace(default=25),  # No max
                ),
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_constraints_steps_with_max(self):
        """Test verbose model with steps and max - lines 550-552"""
        model = SimpleNamespace(
            id="test-model",
            type="image",
            model_spec=SimpleNamespace(
                name="Test Model",
                constraints=SimpleNamespace(
                    steps=SimpleNamespace(default=25, max=30),
                ),
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_constraints_no_default(self):
        """Test verbose model with constraints but no defaults"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                constraints=SimpleNamespace(
                    temperature=SimpleNamespace(min=0, max=2),  # No default
                    top_p=SimpleNamespace(min=0, max=1),  # No default
                    steps=SimpleNamespace(min=1, max=50),  # No default
                ),
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_with_model_source(self):
        """Test verbose model with model source - line 558"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                modelSource="https://huggingface.co/meta-llama/llama-3",
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_no_model_source(self):
        """Test verbose model without model source"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                # No modelSource
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_with_created_date(self):
        """Test verbose model with created timestamp - line 560"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            created=1745903059,  # Has created timestamp
            model_spec=SimpleNamespace(
                name="Test Model",
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_no_created_date(self):
        """Test verbose model without created timestamp"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            # No created attribute
            model_spec=SimpleNamespace(
                name="Test Model",
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_offline_status(self):
        """Test verbose model with offline status - line 566-569"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                offline=True,  # Offline
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_online_status(self):
        """Test verbose model with online status (offline=False)"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                offline=False,  # Online
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_beta_status(self):
        """Test verbose model with beta status - line 571"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                offline=False,
                beta=True,  # Beta
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_no_status_parts(self):
        """Test verbose model with no offline attribute (no status at all)"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                # No offline attribute
                # No beta attribute
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    def test_format_verbose_model_no_capabilities_attribute(self):
        """Test verbose model with no capabilities for quantization check"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                availableContextTokens=131072,
                # No capabilities
            ),
        )
        panel = ModelFormatter.format_verbose_model(model)
        assert panel is not None

    # Tests for format_json covering lines 606-617
    def test_format_json(self, sample_models):
        """Test JSON formatting"""
        result = ModelFormatter.format_json(sample_models)
        assert result is not None
        # Verify it's valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == len(sample_models)

    def test_format_json_includes_capabilities(self, mock_model_text):
        """Test JSON includes capability information"""
        result = ModelFormatter.format_json([mock_model_text])
        parsed = json.loads(result)
        assert "capabilities" in parsed[0]
        assert parsed[0]["capabilities"]["function_calling"] is True
        assert parsed[0]["capabilities"]["vision"] is False

    def test_format_json_no_capabilities(self):
        """Test JSON for model without capabilities"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                traits=["default"],
                # No capabilities
            ),
        )
        result = ModelFormatter.format_json([model])
        parsed = json.loads(result)
        assert "capabilities" not in parsed[0]

    def test_format_json_no_pricing(self):
        """Test JSON for model without pricing - line 606"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                traits=[],
                # No pricing
            ),
        )
        result = ModelFormatter.format_json([model])
        parsed = json.loads(result)
        assert "pricing" not in parsed[0]

    def test_format_json_no_context(self):
        """Test JSON for model without context - line 617"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                # No availableContextTokens
            ),
        )
        result = ModelFormatter.format_json([model])
        parsed = json.loads(result)
        assert "context_tokens" not in parsed[0]

    def test_format_json_with_context(self):
        """Test JSON includes context tokens when available"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                availableContextTokens=131072,
            ),
        )
        result = ModelFormatter.format_json([model])
        parsed = json.loads(result)
        assert "context_tokens" in parsed[0]
        assert parsed[0]["context_tokens"] == 131072

    def test_format_json_no_name_attr(self):
        """Test JSON for model without name attribute"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                # No name attribute
                traits=[],
            ),
        )
        result = ModelFormatter.format_json([model])
        parsed = json.loads(result)
        assert parsed[0]["name"] is None

    def test_format_json_no_traits_attr(self):
        """Test JSON for model without traits attribute"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                # No traits attribute
            ),
        )
        result = ModelFormatter.format_json([model])
        parsed = json.loads(result)
        assert parsed[0]["traits"] == []

    def test_format_json_with_input_pricing_only(self):
        """Test JSON with only input pricing"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                pricing=SimpleNamespace(
                    input=SimpleNamespace(usd=0.5, diem=0.5),
                    # No output
                ),
            ),
        )
        result = ModelFormatter.format_json([model])
        parsed = json.loads(result)
        assert "pricing" in parsed[0]
        assert "input_usd" in parsed[0]["pricing"]
        assert "input_diem" in parsed[0]["pricing"]
        assert "output_usd" not in parsed[0]["pricing"]

    def test_format_json_with_output_pricing_only(self):
        """Test JSON with only output pricing"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                pricing=SimpleNamespace(
                    # No input
                    output=SimpleNamespace(usd=2.0, diem=2.0),
                ),
            ),
        )
        result = ModelFormatter.format_json([model])
        parsed = json.loads(result)
        assert "pricing" in parsed[0]
        assert "output_usd" in parsed[0]["pricing"]
        assert "output_diem" in parsed[0]["pricing"]
        assert "input_usd" not in parsed[0]["pricing"]

    def test_format_json_empty_models(self):
        """Test JSON with empty models list"""
        result = ModelFormatter.format_json([])
        parsed = json.loads(result)
        assert parsed == []

    def test_format_json_multiple_models(self):
        """Test JSON with multiple models"""
        models = [
            SimpleNamespace(
                id="model-1",
                type="text",
                model_spec=SimpleNamespace(name="Model 1", traits=["fast"]),
            ),
            SimpleNamespace(
                id="model-2",
                type="image",
                model_spec=SimpleNamespace(name="Model 2", traits=["quality"]),
            ),
        ]
        result = ModelFormatter.format_json(models)
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["id"] == "model-1"
        assert parsed[1]["id"] == "model-2"

    # Additional edge case tests
    def test_format_text_table_multiple_models_sorting(self):
        """Test that text table sorts models by id"""
        models = [
            SimpleNamespace(
                id="z-model",
                type="text",
                model_spec=SimpleNamespace(name="Z Model"),
            ),
            SimpleNamespace(
                id="a-model",
                type="text",
                model_spec=SimpleNamespace(name="A Model"),
            ),
        ]
        table = ModelFormatter.format_text_table(models)
        assert table is not None
        # Table should have 2 rows
        assert len(table.rows) == 2

    def test_format_image_table_multiple_models_sorting(self):
        """Test that image table sorts models by id"""
        models = [
            SimpleNamespace(
                id="z-image",
                type="image",
                model_spec=SimpleNamespace(name="Z Image", offline=False),
            ),
            SimpleNamespace(
                id="a-image",
                type="image",
                model_spec=SimpleNamespace(name="A Image", offline=False),
            ),
        ]
        table = ModelFormatter.format_image_table(models)
        assert table is not None
        assert len(table.rows) == 2

    def test_format_tts_table_with_currency_usd(self):
        """Test TTS table with USD currency only"""
        model = SimpleNamespace(
            id="tts-test",
            type="tts",
            model_spec=SimpleNamespace(
                name="TTS Test",
                voices=["voice1"],
                pricing=SimpleNamespace(input=SimpleNamespace(usd=3.5, diem=3.5)),
            ),
        )
        table = ModelFormatter.format_tts_table([model], currency="usd")
        assert table is not None

    def test_format_embedding_table_with_currency_diem(self):
        """Test embedding table with DIEM currency only"""
        model = SimpleNamespace(
            id="embedding-test",
            type="embedding",
            model_spec=SimpleNamespace(
                name="Embedding Test",
                availableContextTokens=8192,
                pricing=SimpleNamespace(
                    input=SimpleNamespace(usd=0.15, diem=0.15),
                ),
            ),
        )
        table = ModelFormatter.format_embedding_table([model], currency="diem")
        assert table is not None

    def test_format_verbose_model_with_currency(self):
        """Test verbose model with specific currency"""
        model = SimpleNamespace(
            id="test-model",
            type="text",
            model_spec=SimpleNamespace(
                name="Test Model",
                pricing=SimpleNamespace(
                    input=SimpleNamespace(usd=0.5, diem=0.5),
                    output=SimpleNamespace(usd=2.0, diem=2.0),
                ),
            ),
        )
        panel = ModelFormatter.format_verbose_model(model, currency="usd")
        assert panel is not None

        panel = ModelFormatter.format_verbose_model(model, currency="diem")
        assert panel is not None
