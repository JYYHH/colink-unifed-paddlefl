#!/usr/bin/env python

from setuptools import setup, find_namespace_packages

FRAMEWORK_NAME = 'paddlefl'

setup(name=f'colink-unifed-{FRAMEWORK_NAME}',
      version='0.0',
      packages=find_namespace_packages(
          'src', exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      package_dir={'': 'src'},
      install_requires=[
          'colink >= 0.3.0',
          "flbenchmark",
      ],
      entry_points={
          'console_scripts': [
              f'unifed-{FRAMEWORK_NAME} = unifed.frameworks.{FRAMEWORK_NAME}:run_protocol',
          ] + ([f'unifed-{FRAMEWORK_NAME}-workload = unifed.frameworks.{FRAMEWORK_NAME}:simulate_workload'] if FRAMEWORK_NAME != 'example' else ["unifed-example-workload = unifed.frameworks.example:simulate_workload"]),
      }
      )
