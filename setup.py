import setuptools

if __name__ == '__main__':
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setuptools.setup(
        version='1.0.0',
        # author_email='',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url='https://github.com/Yukiya-Umimi/ITUNet-for-PICAI-2022-Challenge',
        project_urls={
            "Bug Tracker": "https://github.com/Yukiya-Umimi/ITUNet-for-PICAI-2022-Challenge/issues"
        },
        license='Apache 2.0',
        packages=setuptools.find_packages('.', exclude=['tests']),
    )
