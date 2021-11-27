# submit predictions to kaggle

```bash
# set environment variables KAGGLE_USERNAME and KAGGLE_KEY
source private.env
echo $KAGGLE_USERNAME
echo $KAGGLE_KEY

# show competitions
kaggle competitions list

# submit file
FILE=/tmp/working/data/submission_test.csv 
SUBMISSION_NAME="Submission where pressure=2*u_in"
COMPETITION_NAME=ventilator-pressure-prediction

!kaggle competitions submit -f $FILE -m $SUBMISSION_NAME $COMPETITION_NAME
```

# start docker

```bash
docker images list

CONTAINER_NAME=kaggle_for_vscode_v3-works
docker container start $CONTAINER_NAME
docker container list
```

# Rewrite git history

https://docs.github.com/en/repositories/working-with-files/managing-large-files/removing-files-from-git-large-file-storage

### using git-filter-repo


```bash
# try git-filter-repo
sudo pip install git-filter-repo

# install git>=2.22.0, which is not available from repositoryh
sudo add-apt-repository ppa:git-core/ppa -y
sudo apt-get update
sudo apt-get install git -y
git --version

# filter all files >30MB
sudo git-filter-repo --strip-blobs-bigger-than 30M --debug --dry-run

# filter all CSV files
sudo git-filter-repo --path-glob '.csv'
```

### using BFG

```bash
# test installation
BFG_PATH=/mnt/sda1/projects/git/_archive/bfg-1.14.0.jar
alias bfg="java -jar $BFG_PATH"
bfg

# usage is bfg [options] [<repo>]
REPO_PATH=/mnt/sda1/projects/git/kaggle/ventilator_pressure_prediction
bfg --strip-blobs-bigger-than 30M $REPO_PATH
```