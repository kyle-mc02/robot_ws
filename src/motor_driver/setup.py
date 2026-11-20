from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'motor_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (os.path.join('share', package_name, 'maps'), glob('maps/*.yaml') + glob('maps/*.pgm')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='koil',
    maintainer_email='koil@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'motor_gui_node = motor_driver.motor_gui_node:main',
            'driver_node = motor_driver.driver_node:main',
            'controller_node = motor_driver.controller_node:main',
            'command_router = motor_driver.command_router:main',
            'perception = motor_driver.perception:main',
        ],
    },
)
