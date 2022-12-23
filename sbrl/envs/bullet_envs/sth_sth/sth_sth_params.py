"""
Sample environment params w/ task labels
"""

from sbrl.envs.bullet_envs.sth_sth import BulletSthSth
from sbrl.utils.python_utils import AttrDict as d

###### COMMON ######

# panda + gripper + single item
panda_gripper_single = d(
    
)

###### PARAMS ######

# 5: Closing something

# 6: Covering something with something

# 7: Digging something out of something

# 8: Dropping something behind something

# 9: Dropping something in front of something

# 10: Dropping something into something

if __name__ == '__main__':
    testParams = None
    BulletSthSth()
