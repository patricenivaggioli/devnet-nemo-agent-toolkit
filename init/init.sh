#!/bin/bash

#executed at deployment for any VM customization

ETIPATH="/usr/share/etilabs"

CNT=`head -n 1 $ETIPATH/details | tail -1`
LABNAME=`head -n 2 $ETIPATH/details | tail -1`
LABURL=`head -n 3 $ETIPATH/details | tail -1`

#in case there is some additional file with keys, accounts, read the $CNT line. File should be in this same directory
#LINE=`head -n $CNT $ETIPATH/labinit/update/content/init/accounts | tail -1`
#set $LINE
#rm -f $ETIPATH/labinit/update/content/init/accounts

# additional nginx config - if any reverse proxy needs to get configured
#sudo cp -f $ETIPATH/labinit/update/content/init/nginx_example /etc/nginx/sites-enabled
#sudo nginx -s reload

# ...
#sudo apt-get update && sudo apt-get install -y gpg lsb-release

# jupyter is not started by default, if you need it start it with
sudo systemctl enable jupyter
sudo service jupyter start

