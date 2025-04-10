# this is setup.py
import setuptools

setuptools.setup(
        name="mae_tf2",
        version="0.2.10.0",
        author="vwhvpwvk",
        description="TensorFlow 2.0 implementation of MAE",
        packages=setuptools.find_packages(),
        install_requires=[
            'tensorflow==2.10.*'
            ]
)

