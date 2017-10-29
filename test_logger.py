

from gym_logger import GymLogger


meta_data = {}
meta_data['a'] = 523
meta_data['b'] = 22

logger = GymLogger(meta_data, value_names=('x', 'y', 'z'))


logger.update(('a', 44, -0.1))
logger.update(('b', 22, -0.5))

logger.save_history()


