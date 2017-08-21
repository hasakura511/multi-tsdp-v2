# multi-tsdp

## Installation

### Linux
```
git clone git@github.com:hasakura511/multi-tsdp.git
docker-compose build
docker-compose run backend python manage.py migrate
```

### Mac
```
brew install docker docker-compose docker-machine xhyve docker-machine-driver-xhyve
docker-machine create --driver xhyve --xhyve-experimental-nfs-share 
docker-machine start default
eval $(docker-machine env default)
docker-compose build

# Install SSHFS on VM
docker-machine ssh

    # Run these commands on VM
    $ tce-load -wi sshfs-fuse
    $ mkdir multi-tsdp
    $ sudo sshfs <username>@$<IP>:/<tsdp_dir>/ /home/docker/multi-tsdp/

docker-compose run --volume=/home/docker/multi-tsdp/backend/src/:/srv/app backend python manage.py migrate
docker-compose run -p 8000:8000 -d --volume=/home/docker/multi-tsdp/backend/src/:/srv/app backend
# get IP from VM, use one in the browser <IP>:8000
echo $DOCKER_HOST

```

## Configuration

### Backend

tsdp.settings.py:
```
IB_HOST = '127.0.0.1'
IB_PORT = 4001
IB_CLIENT_ID = '999'

#  Non-mandatory settings (have default values) 
IB_MAX_WAIT_SECONDS = 30
```

### IB Gateway