# Mushroom Model API

#############################
# Prerequisites
#############################
Install Python
Install VSCode or IDE of choice
Homebrew:
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```
pipenv:
```
brew install pipenv
```

#############################
# Initial Project Setup
#############################
1) Add .gitignore and Readme.md file
2) Pipfile - add basic python environment to get started, no packages defined
3) Pipfile.lock - start with empty {} file

#############################
# Python Environment Setup
#############################
# To activate virtualenv
```
pipenv shell
pipenv clean
```


###################################
# Adding Pip packages (Only required for brand new Pipfile)
###################################
pipenv lock --clear
pipenv install tensorflow
pipenv install fastapi
pipenv install uvicorn
pipenv install aiofiles
pipenv install opencv-python
pipenv install python-multipart

#############################
# Client API Service
#############################
Run the API Service by:
```
uvicorn server:app
```

# Tunnel your app on localhost to public access
```
brew install agrinman/tap/tunnelto
```

# login with your account key
```
$ tunnelto set-auth --key xxxxxxxxxx
```

# start the tunnel
```
$ tunnelto --subdomain mushroom --port 8000
```



 
