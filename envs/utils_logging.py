import csv
import os
import time

class EpisodeLogger:
    def __init__(self, log_dir, episode_id):
        self.log_dir = log_dir
        self.episode_id = episode_id
        self.filename = os.path.join(log_dir, f"episode_{episode_id}.csv")
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        
        # Header
        self.header = [
            'time', 'step', 'agent_id',
            'px_enu', 'py_enu', 'pz_enu',
            'vx_enu', 'vy_enu', 'vz_enu',
            'yaw_enu',
            'gx_enu', 'gy_enu', 'gz_enu',
            'ax_body', 'ay_body', 'az_enu', 'yaw_rate',
            'reward', 'collision', 'goal_reached', 'sep_min'
        ]
        self.writer.writerow(self.header)
        
    def log_step(self, step, timestamp, agent_data):
        """
        agent_data: dict of agent_id -> dict of features
        """
        for agent_id, data in agent_data.items():
            row = [
                timestamp, step, agent_id,
                data.get('px', 0), data.get('py', 0), data.get('pz', 0),
                data.get('vx', 0), data.get('vy', 0), data.get('vz', 0),
                data.get('yaw', 0),
                data.get('gx', 0), data.get('gy', 0), data.get('gz', 0),
                data.get('ax', 0), data.get('ay', 0), data.get('az', 0), data.get('yaw_rate', 0),
                data.get('reward', 0),
                data.get('collision', False),
                data.get('goal_reached', False),
                data.get('sep_min', -1)
            ]
            self.writer.writerow(row)
            
    def close(self):
        self.file.close()

class BatchLogger:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.current_logger = None
        
    def start_episode(self, episode_id):
        if self.current_logger:
            self.current_logger.close()
        self.current_logger = EpisodeLogger(self.data_dir, episode_id)
        
    def log_step(self, step, timestamp, agent_data):
        if self.current_logger:
            self.current_logger.log_step(step, timestamp, agent_data)
            
    def end_episode(self):
        if self.current_logger:
            self.current_logger.close()
            self.current_logger = None
