from custom_environment_p import SimpleGridWorld
import pettingzoo
from pettingzoo.test import parallel_api_test
import numpy.random as npr
import logging
logger = logging.getLogger(__name__)

gridworld = (SimpleGridWorld())
gridworld.reset()
gridworld.render()
logging.basicConfig(filename='rand_agent_actions.log', level=logging.INFO)
logger.info('Started')
for i in range(1_000_000_000_000) :
    gridworld.reset()
    for j in range (100) :
        action=npr.randint(low=0, high=4)
        gridworld.step(actions={"agent_0":action})
    logger.info(gridworld.final_cost())

