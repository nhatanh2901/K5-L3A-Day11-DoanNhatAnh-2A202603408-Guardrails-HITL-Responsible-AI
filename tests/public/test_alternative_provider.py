"""Verify the explicitly selected alternative keeps both targets on one API."""
from core import config


def test_groq_key_alias_and_honest_provider_selection(monkeypatch):
    monkeypatch.setenv("RED_TEAM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROUQ_API_KEY", "test-groq-dummy")
    assert config.get_red_provider() == "groq"
    assert config.red_uses_openai_sdk()
    assert not config.uses_rubric_default_model()
    kwargs = config.red_openai_client_kwargs()
    assert kwargs["api_key"] == "test-groq-dummy"
    assert kwargs["base_url"] == "https://api.groq.com/openai/v1"
    assert config.get_blue_model() == "liquid/lfm-3-1.7b:free"


def test_groq_target_factories_preserve_nemo(monkeypatch):
    from agents.agent import create_red_agent_default, RED_DEFAULT_INSTRUCTION
    from agents.guards_agent import create_red_agent_advance, RedAdvanceGuardPlugin
    from agents.red_advance_nemo import build_rails_dict, NemoRailsGuard

    monkeypatch.setenv("RED_TEAM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-dummy")
    agent, runner = create_red_agent_default()
    assert agent.instruction == RED_DEFAULT_INSTRUCTION
    _, advance_runner = create_red_agent_advance()
    assert runner.provider == advance_runner.provider == "groq"
    assert runner.model == advance_runner.model == "openai/gpt-oss-20b"
    plugin = advance_runner.plugins[0]
    assert isinstance(plugin, RedAdvanceGuardPlugin)
    assert isinstance(plugin.nemo, NemoRailsGuard)
    entry = build_rails_dict()["models"][0]
    assert entry["engine"] == "openai"
    assert entry["api_key_env_var"] == "GROQ_API_KEY"
    assert entry["parameters"]["base_url"] == runner.client_kwargs["base_url"]
