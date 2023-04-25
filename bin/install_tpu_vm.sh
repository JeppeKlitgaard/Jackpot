#!/bin/bash
set +xe

# Install Pyenv
curl https://pyenv.run | bash
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
