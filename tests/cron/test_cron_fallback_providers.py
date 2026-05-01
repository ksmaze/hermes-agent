"""Tests for per-job fallback_providers support in cron jobs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestCreateJobFallbackProviders:
    def test_create_job_persists_normalized_fallback_providers(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job

        job = create_job(
            prompt="hello",
            schedule="every 1h",
            fallback_providers=[
                {
                    "provider": " custom-provider ",
                    "model": " model-a ",
                    "base_url": "http://localhost:9000/v1/",
                    "api_key": " secret ",
                    "key_env": " TEST_KEY ",
                }
            ],
        )
        stored = get_job(job["id"])

        assert stored["fallback_providers"] == [
            {
                "provider": "custom-provider",
                "model": "model-a",
                "base_url": "http://localhost:9000/v1",
                "api_key": "secret",
                "key_env": "TEST_KEY",
            }
        ]

    def test_create_job_without_fallbacks_omits_field(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job

        job = create_job(prompt="hello", schedule="every 1h")
        stored = get_job(job["id"])

        assert "fallback_providers" not in stored

    def test_create_job_empty_fallbacks_preserves_explicit_override(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job

        job = create_job(
            prompt="hello",
            schedule="every 1h",
            fallback_providers=[],
        )
        stored = get_job(job["id"])

        assert stored["fallback_providers"] == []


class TestCronjobToolFallbackProviders:
    def test_create_and_update_round_trip(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob

        created = json.loads(
            cronjob(
                action="create",
                prompt="hello",
                schedule="every 1h",
                fallback_providers=[
                    {"provider": "job-provider", "model": "job-model"}
                ],
            )
        )
        assert created["success"] is True
        assert created["job"]["fallback_providers"] == [
            {"provider": "job-provider", "model": "job-model"}
        ]

        updated = json.loads(
            cronjob(
                action="update",
                job_id=created["job_id"],
                fallback_providers=[],
            )
        )
        assert updated["success"] is True
        assert updated["job"]["fallback_providers"] == []

        listing = json.loads(cronjob(action="list"))
        assert listing["jobs"][0]["fallback_providers"] == []


class TestRunJobFallbackProviders:
    @staticmethod
    def _install_stubs(monkeypatch, observed: dict):
        import cron.scheduler as sched

        class FakeAgent:
            def __init__(self, **kwargs):
                observed["fallback_model"] = kwargs.get("fallback_model")

            def run_conversation(self, *_a, **_kw):
                return {"final_response": "done", "messages": []}

            def get_activity_summary(self):
                return {"seconds_since_activity": 0.0}

        fake_mod = type(sys)("run_agent")
        fake_mod.AIAgent = FakeAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_mod)

        from hermes_cli import runtime_provider as _rtp

        monkeypatch.setattr(
            _rtp,
            "resolve_runtime_provider",
            lambda **_kw: {
                "provider": "primary",
                "api_key": "k",
                "base_url": "http://primary.local/v1",
                "api_mode": "chat_completions",
            },
        )

        monkeypatch.setattr(sched, "_build_job_prompt", lambda job, prerun_script=None: "hi")
        monkeypatch.setattr(sched, "_resolve_origin", lambda job: None)
        monkeypatch.setattr(sched, "_resolve_delivery_target", lambda job: None)
        monkeypatch.setattr(sched, "_resolve_cron_enabled_toolsets", lambda job, cfg: None)
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")

        import dotenv

        monkeypatch.setattr(dotenv, "load_dotenv", lambda *_a, **_kw: True)

    def test_run_job_uses_job_specific_fallbacks_over_global(self, tmp_path, monkeypatch):
        import cron.scheduler as sched

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)
        monkeypatch.setattr(sched, "_hermes_home", tmp_path)
        (tmp_path / "config.yaml").write_text(
            "fallback_providers:\n"
            "  - provider: global-provider\n"
            "    model: global-model\n",
            encoding="utf-8",
        )

        job = {
            "id": "job1",
            "name": "job1",
            "schedule_display": "manual",
            "fallback_providers": [
                {"provider": "job-provider", "model": "job-model"}
            ],
        }

        success, *_ = sched.run_job(job)
        assert success is True
        assert observed["fallback_model"] == [
            {"provider": "job-provider", "model": "job-model"}
        ]

    def test_run_job_empty_job_fallbacks_disable_global(self, tmp_path, monkeypatch):
        import cron.scheduler as sched

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)
        monkeypatch.setattr(sched, "_hermes_home", tmp_path)
        (tmp_path / "config.yaml").write_text(
            "fallback_providers:\n"
            "  - provider: global-provider\n"
            "    model: global-model\n",
            encoding="utf-8",
        )

        job = {
            "id": "job2",
            "name": "job2",
            "schedule_display": "manual",
            "fallback_providers": [],
        }

        success, *_ = sched.run_job(job)
        assert success is True
        assert observed["fallback_model"] == []
