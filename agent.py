import importlib
import torch

def make_agent_from_source(path, *, device, parallel=1):
	spec = importlib.util.spec_from_file_location('dynamic_agent', path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	Agent = getattr(module, 'Agent')
	agent = Agent(device=device, parallel=parallel).to(device)
	agent.source_path = path
	agent.parallel = parallel
	agent.step_count = 0
	agent.session_count = 0
	agent.log = []
	return agent

def load_agent_from_save(path, *, device, parallel=1, strict=False):
    save = torch.load(path, map_location=device, weights_only=False)
    source_path = save.get('source_path') or save.get('source_path')
    agent = make_agent_from_source(source_path, device=device)
    agent.load_state_dict(save['agent_state'], strict=strict)
    agent.step_count = save.get('step_count', 0)
    agent.session_count = save.get('session_count', 0) + 1
    agent.log = save.get('log', [])
    return agent

def load_agent(save_path, source_path, device):
	try:
		agent = load_agent_from_save(save_path, device=device)
	except FileNotFoundError as error:
		print('NEW AGENT')
		agent = make_agent_from_source(source_path, device=device)
	except Exception as error:
		print('BAD SAVE', save_path)
		raise error
