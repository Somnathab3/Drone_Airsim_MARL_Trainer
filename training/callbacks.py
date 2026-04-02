from ray.rllib.algorithms.callbacks import DefaultCallbacks
import numpy as np

class UAveMetricsCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, episode, **kwargs):
        # Initialize custom metrics using new API
        pass  # Metrics will be set at episode end
        
    def on_episode_step(self, *, episode, **kwargs):
        # We can aggregate info from agents
        # env = base_env.get_sub_environments()[env_index]
        # But for ParallelEnv, wrapper might hide it.
        # Efficient way: access 'last_info' if available.
        # Or let Env log cumulative stats to info['episode_metrics'] and read it at end?
        pass

    def on_episode_end(self, *, episode, **kwargs):
        # Aggregate agent infos
        # episode.last_info_for(agent_id) gives info dict
        
        # Check all agents
        successes = []
        collisions = []
        min_seps = []
        resolutions = []
        deadlocks = []
        team_progress = []
        
        # Get agent IDs from episode
        try:
            agent_ids = episode.get_agents()
        except AttributeError:
            agent_ids = episode.agent_ids if hasattr(episode, 'agent_ids') else []
        
        for agent_id in agent_ids:
            try:
                last_info = episode.last_info_for(agent_id)
            except (AttributeError, KeyError):
                continue
                
            if not last_info:
                continue
                
            outcome = last_info.get('outcome', 'timeout')
            successes.append(1.0 if outcome == 'success' else 0.0)
            collisions.append(1.0 if outcome == 'collision' else 0.0)
            
            sep = last_info.get('sep_min', float('inf'))
            if sep != float('inf'):
                min_seps.append(sep)
            
            # Cooperative Metrics
            resolutions.append(1.0 if last_info.get('resolution_active', False) else 0.0)
            deadlocks.append(1.0 if last_info.get('r_deadlock_penalty', 0) < 0 else 0.0)
            team_progress.append(last_info.get('r_team_progress', 0.0))
            
        # Set custom metrics using new API
        if successes:
            metrics = {
                "success_rate": np.mean(successes),
                "collision_rate": np.mean(collisions),
                "min_separation": np.min(min_seps) if min_seps else 0.0,
                "team_success_rate": 1.0 if any(s > 0.5 for s in successes) else 0.0,
                "conflict_resolution_count": np.sum(resolutions),
                "deadlock_rate": np.mean(deadlocks),
                "team_progress_mean": np.mean(team_progress)
            }
            # Try new API first, fall back to old if needed
            if hasattr(episode, 'set_custom_metrics'):
                episode.set_custom_metrics(metrics)
            elif hasattr(episode, 'custom_metrics'):
                episode.custom_metrics.update(metrics)
