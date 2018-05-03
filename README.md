$ # install the utility
$ pip install git+https://github.com/gldnspud/virtualenv-pythonw-osx.git
$ # enter the virtualenv with virtualenvwrapper (or manually)
$ workon my-venv
$ # double-check that this is your venv Python binary
$ which python
/Users/macbook/.virtualenvs/my-venv/bin/python
$ # fix it, using magic
$ fix-osx-virtualenv `which python`/../..

