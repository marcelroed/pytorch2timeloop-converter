from setuptools import setup, find_packages

setup(name='pytorch2timeloop',
        version='0.2',
        url='https://github.com/Accelergy-Project/pytorch2timeloop-converter',
        license='MIT',
        install_requires=[
            "torch",
            "torchvision",
            "numpy",
            "pyyaml",
            "transformers"
        ],
        python_requires='>=3.10',
        include_package_data=True,
        packages=find_packages())
