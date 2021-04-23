# Mushroom Downloader

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

# Chrome driver
# Download driver based one current Chrome version

#run this in the driver folder:
```
xattr -d com.apple.quarantine chromedriver
```

#############################
# Python Environment Setup
#############################
# To activate virtualenv
```
pipenv shell
```


# Install from piplock file
```
pipenv install
```

###################################
# Adding Pip packages
###################################
```
pipenv lock --clear
pipenv install selenium
pipenv install user_agent
pipenv install requests
```

Updating packages
```
pipenv update
```


# Run the image crawler
```
python -m test_crawler
```


 
