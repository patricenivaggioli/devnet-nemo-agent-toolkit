#!/bin/bash

#replace custom strings in lab, this gets executed on every lab update

ETIPATH="/usr/share/etilabs"

#these are set/saved by the deployer in /usr/share/etilabs/details
CNT=`head -n 1 $ETIPATH/details | tail -1`
LABNAME=`head -n 2 $ETIPATH/details | tail -1`
LABURL=`head -n 3 $ETIPATH/details | tail -1`

TWODIG=$CNT
if [ ${#TWODIG} -lt 2 ] ; then
  TWODIG="0"$TWODIG
fi

#lab title - if you need to identify each instance in the title
#sed -i -e "s/%%LABID%%/Station $TWODIG/g" /var/www/html/lab/meta/metadata

#labcontent - replace custom placeholders in the lab content
#find /var/www/html/lab -type f -exec sed -i "s|%%LABNAME%%|$LABNAME|g" {} \;
#find /var/www/html/lab -type f -exec sed -i "s|%%LABURL%%|$LABURL|g" {} \;


#lab prompt - if you need to identify each instance in the terminal prompt
#sed -i -e "s#outshift>#outshift-lab$TWODIG>#g" /home/ubuntu/.bashrc
