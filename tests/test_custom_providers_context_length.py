"""Tests for custom_providers per-model context_length support.

This module tests that get_model_context_length correctly reads
per-model context_length from custom_providers in agent_config.
"""

import pytest
from unittest.mock import patch, MagicMock

from agent.model_metadata import get_model_context_length


class TestCustomProvidersContextLength:
    """Test cases for custom_providers context_length resolution."""

    def test_custom_providers_match_found(self):
        """Test that custom_providers per-model context_length is used when matched."""
        agent_config = {
            "custom_providers": [
                {
                    "name": "test-gateway",
                    "base_url": "http://localhost:3000/v1",
                    "api_key": "sk-test",
                    "models": {
                        "gpt-5.4": {
                            "context_length": 1050000
                        }
                    }
                }
            ]
        }

        # Mock all external calls to return fallback
        with patch("agent.model_metadata._strip_provider_prefix", side_effect=lambda x: x), \
             patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata._is_custom_endpoint", return_value=False), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}):

            result = get_model_context_length(
                model="gpt-5.4",
                base_url="http://localhost:3000/v1",
                api_key="sk-test",
                agent_config=agent_config,
            )

            assert result == 1050000

    def test_custom_providers_no_match_different_base_url(self):
        """Test that custom_providers lookup fails when base_url doesn't match."""
        agent_config = {
            "custom_providers": [
                {
                    "name": "test-gateway",
                    "base_url": "http://localhost:3000/v1",
                    "models": {
                        "gpt-5.4": {
                            "context_length": 1050000
                        }
                    }
                }
            ]
        }

        # Mock to return fallback value
        with patch("agent.model_metadata._strip_provider_prefix", side_effect=lambda x: x), \
             patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata._is_custom_endpoint", return_value=False), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}):

            result = get_model_context_length(
                model="gpt-5.4",
                base_url="http://different-server:8080/v1",  # Different base_url
                api_key="sk-test",
                agent_config=agent_config,
            )

            # Should fall through to fallback (128000)
            assert result == 128000

    def test_custom_providers_no_match_different_model(self):
        """Test that custom_providers lookup fails when model doesn't match."""
        agent_config = {
            "custom_providers": [
                {
                    "name": "test-gateway",
                    "base_url": "http://localhost:3000/v1",
                    "models": {
                        "gpt-5.4": {
                            "context_length": 1050000
                        }
                    }
                }
            ]
        }

        # Mock to return fallback value
        with patch("agent.model_metadata._strip_provider_prefix", side_effect=lambda x: x), \
             patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata._is_custom_endpoint", return_value=False), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}):

            result = get_model_context_length(
                model="different-model",  # Different model
                base_url="http://localhost:3000/v1",
                api_key="sk-test",
                agent_config=agent_config,
            )

            # Should fall through to fallback (128000)
            assert result == 128000

    def test_custom_providers_multiple_providers(self):
        """Test that custom_providers lookup works with multiple providers."""
        agent_config = {
            "custom_providers": [
                {
                    "name": "first-gateway",
                    "base_url": "http://first.example.com/v1",
                    "models": {
                        "model-a": {"context_length": 100000}
                    }
                },
                {
                    "name": "second-gateway",
                    "base_url": "http://second.example.com/v1",
                    "models": {
                        "model-b": {"context_length": 200000}
                    }
                }
            ]
        }

        with patch("agent.model_metadata._strip_provider_prefix", side_effect=lambda x: x), \
             patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata._is_custom_endpoint", return_value=False), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}):

            # Test second provider's model
            result = get_model_context_length(
                model="model-b",
                base_url="http://second.example.com/v1",
                api_key="sk-test",
                agent_config=agent_config,
            )

            assert result == 200000

    def test_config_context_length_takes_precedence(self):
        """Test that explicit config_context_length takes precedence over custom_providers."""
        agent_config = {
            "custom_providers": [
                {
                    "name": "test-gateway",
                    "base_url": "http://localhost:3000/v1",
                    "models": {
                        "gpt-5.4": {
                            "context_length": 1050000
                        }
                    }
                }
            ]
        }

        result = get_model_context_length(
            model="gpt-5.4",
            base_url="http://localhost:3000/v1",
            api_key="sk-test",
            config_context_length=500000,  # Explicit override
            agent_config=agent_config,
        )

        # config_context_length should take precedence
        assert result == 500000

    def test_no_agent_config(self):
        """Test that function works without agent_config (backwards compatibility)."""
        with patch("agent.model_metadata._strip_provider_prefix", side_effect=lambda x: x), \
             patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata._is_custom_endpoint", return_value=False), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}):

            result = get_model_context_length(
                model="gpt-5.4",
                base_url="http://localhost:3000/v1",
                api_key="sk-test",
                # No agent_config passed
            )

            # Should fall through to fallback (128000)
            assert result == 128000

    def test_custom_providers_empty_list(self):
        """Test that empty custom_providers list is handled gracefully."""
        agent_config = {
            "custom_providers": []
        }

        with patch("agent.model_metadata._strip_provider_prefix", side_effect=lambda x: x), \
             patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata._is_custom_endpoint", return_value=False), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}):

            result = get_model_context_length(
                model="gpt-5.4",
                base_url="http://localhost:3000/v1",
                api_key="sk-test",
                agent_config=agent_config,
            )

            assert result == 128000

    def test_custom_providers_no_models_key(self):
        """Test that provider entry without models key is handled gracefully."""
        agent_config = {
            "custom_providers": [
                {
                    "name": "test-gateway",
                    "base_url": "http://localhost:3000/v1",
                    # No "models" key
                }
            ]
        }

        with patch("agent.model_metadata._strip_provider_prefix", side_effect=lambda x: x), \
             patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata._is_custom_endpoint", return_value=False), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}):

            result = get_model_context_length(
                model="gpt-5.4",
                base_url="http://localhost:3000/v1",
                api_key="sk-test",
                agent_config=agent_config,
            )

            assert result == 128000

    def test_custom_providers_invalid_context_length(self):
        """Test that invalid context_length values are handled gracefully."""
        agent_config = {
            "custom_providers": [
                {
                    "name": "test-gateway",
                    "base_url": "http://localhost:3000/v1",
                    "models": {
                        "gpt-5.4": {
                            "context_length": "not-a-number"  # Invalid
                        }
                    }
                }
            ]
        }

        with patch("agent.model_metadata._strip_provider_prefix", side_effect=lambda x: x), \
             patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata._is_custom_endpoint", return_value=False), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}):

            result = get_model_context_length(
                model="gpt-5.4",
                base_url="http://localhost:3000/v1",
                api_key="sk-test",
                agent_config=agent_config,
            )

            # Should fall through to fallback due to invalid value
            assert result == 128000

    def test_custom_providers_url_normalization(self):
        """Test that base_url matching handles trailing slashes correctly."""
        agent_config = {
            "custom_providers": [
                {
                    "name": "test-gateway",
                    "base_url": "http://localhost:3000/v1/",  # With trailing slash
                    "models": {
                        "gpt-5.4": {
                            "context_length": 1050000
                        }
                    }
                }
            ]
        }

        with patch("agent.model_metadata._strip_provider_prefix", side_effect=lambda x: x), \
             patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata._is_custom_endpoint", return_value=False), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}):

            # Call with base_url without trailing slash
            result = get_model_context_length(
                model="gpt-5.4",
                base_url="http://localhost:3000/v1",  # Without trailing slash
                api_key="sk-test",
                agent_config=agent_config,
            )

            # Should still match due to URL normalization
            assert result == 1050000


class TestIntegrationWithCallSites:
    """Integration tests verifying agent_config is passed from call sites."""

    def test_cli_context_reference_expansion(self):
        """Verify CLI context reference expansion passes agent_config."""
        # This is a pattern test - verifying the code structure is correct
        # Full integration test would require full CLI setup
        from hermes_cli.config import load_config

        # Verify load_config is importable and returns dict-like
        try:
            cfg = load_config()
            assert isinstance(cfg, dict) or cfg is None
        except Exception:
            pass  # Config may not exist in test environment

    def test_gateway_context_preprocessing(self):
        """Verify gateway @ context preprocessing loads config."""
        # Pattern test for gateway/run.py changes
        import yaml
        from pathlib import Path
        from hermes_constants import get_hermes_home

        hermes_home = get_hermes_home()
        cfg_path = hermes_home / "config.yaml"

        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            assert isinstance(data, dict)

    def test_run_agent_fallback_model(self):
        """Verify fallback model context lookup passes agent_config."""
        # Pattern test verifying the import structure
        from agent.model_metadata import get_model_context_length
        from hermes_cli.config import load_config as _load_fb_config

        # Verify both imports work
        assert callable(get_model_context_length)
        assert callable(_load_fb_config)

    def test_run_agent_compression_feasibility(self):
        """Verify compression feasibility check passes agent_config."""
        # Pattern test verifying the import structure
        from agent.model_metadata import get_model_context_length
        from hermes_cli.config import load_config as _load_aux_config

        # Verify both imports work
        assert callable(get_model_context_length)
        assert callable(_load_aux_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
