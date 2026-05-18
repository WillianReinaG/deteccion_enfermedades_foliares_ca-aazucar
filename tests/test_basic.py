def test_import_agent():
    from app.agent.agent import SugarCaneAgent
    agent = SugarCaneAgent()
    assert agent is not None
